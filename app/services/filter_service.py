import magic
import json
from groq import Groq
from fastapi import UploadFile
from app.core.config import settings
from app.core.schemas import FilterResult, FilterStatus, RejectionReason

client = Groq(api_key=settings.GROQ_API_KEY)


def _call_groq(prompt: str, max_tokens: int = 200) -> str:
    """Single helper for all Groq calls — keeps things DRY."""
    response = client.chat.completions.create(
        model=settings.CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,  # deterministic — important for classifiers
    )
    return response.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    """Strips markdown fences and parses JSON safely."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────
# Layer 1: Hard Filter
# ─────────────────────────────────────────
async def hard_filter(file: UploadFile, content: bytes) -> FilterResult | None:
    """Validates file type and size. No LLM — instant."""

    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        return FilterResult(
            status=FilterStatus.REJECTED,
            reason=RejectionReason.INVALID_TYPE,
            message=f"File type '.{ext}' not supported. Allowed: pdf, docx, txt",
        )

    mime = magic.from_buffer(content[:2048], mime=True)
    expected_mime = settings.ALLOWED_MIME_TYPES.get(ext, "")
    if ext != "txt" and mime != expected_mime:
        return FilterResult(
            status=FilterStatus.REJECTED,
            reason=RejectionReason.INVALID_TYPE,
            message="File content does not match its extension",
        )

    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        return FilterResult(
            status=FilterStatus.REJECTED,
            reason=RejectionReason.TOO_LARGE,
            message=f"File size {size_mb:.1f}MB exceeds the {settings.MAX_FILE_SIZE_MB}MB limit",
        )

    return None


# ─────────────────────────────────────────
# Layer 2: Content Classifier
# ─────────────────────────────────────────
async def content_classifier(text_preview: str, word_count: int) -> FilterResult | None:
    """Uses Groq (Llama 3.3) to classify if content is educational."""

    if word_count < settings.MIN_WORD_COUNT:
        return FilterResult(
            status=FilterStatus.REJECTED,
            reason=RejectionReason.TOO_SHORT,
            message=f"Document too short ({word_count} words). Minimum is {settings.MIN_WORD_COUNT}.",
            word_count=word_count,
        )

    preview = " ".join(text_preview.split()[:800])

    prompt = f"""Analyze this document excerpt and respond ONLY with a JSON object, no extra text.

Document excerpt:
\"\"\"
{preview}
\"\"\"

Respond with exactly this JSON structure:
{{
  "is_educational": true or false,
  "topic": "brief topic in English (max 5 words)",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explanation"
}}

Consider educational: academic papers, textbooks, articles, tutorials, research, reports, study materials.
Consider NOT educational: chat logs, spam, random text, personal diaries, shopping lists."""

    raw = _call_groq(prompt, max_tokens=200)
    result = _parse_json(raw)

    if not result.get("is_educational", False) or result.get("confidence", 0) < 0.5:
        return FilterResult(
            status=FilterStatus.REJECTED,
            reason=RejectionReason.NOT_EDUCATIONAL,
            message=f"Content does not appear to be educational. {result.get('reason', '')}",
            word_count=word_count,
            detected_topic=result.get("topic"),
            confidence=result.get("confidence"),
        )

    return None


async def get_classification(text_preview: str) -> dict:
    """Gets topic + confidence after content passes — used downstream."""
    preview = " ".join(text_preview.split()[:800])

    prompt = f"""Analyze this document excerpt and respond ONLY with a JSON object.

Document excerpt:
\"\"\"
{preview}
\"\"\"

Respond with exactly this JSON:
{{
  "is_educational": true or false,
  "topic": "brief topic in English (max 5 words)",
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explanation"
}}"""

    raw = _call_groq(prompt, max_tokens=200)
    return _parse_json(raw)


# ─────────────────────────────────────────
# Layer 3: Safety Filter
# ─────────────────────────────────────────
async def safety_filter(text_preview: str) -> FilterResult | None:
    """Uses Groq (Llama 3.3) to check for harmful content."""

    preview = " ".join(text_preview.split()[:600])

    prompt = f"""Review this document excerpt for harmful content and respond ONLY with JSON.

Document excerpt:
\"\"\"
{preview}
\"\"\"

Respond with exactly this JSON:
{{
  "is_safe": true or false,
  "reason": "one sentence explanation"
}}

Flag as unsafe ONLY if content contains: hate speech, violence instructions, illegal activities, explicit adult content, self-harm instructions.
Educational content about sensitive topics (history, medicine, law) is safe."""

    raw = _call_groq(prompt, max_tokens=100)
    result = _parse_json(raw)

    if not result.get("is_safe", True):
        return FilterResult(
            status=FilterStatus.REJECTED,
            reason=RejectionReason.UNSAFE_CONTENT,
            message=f"Content flagged as unsafe. {result.get('reason', '')}",
        )

    return None


# ─────────────────────────────────────────
# Main Filter Pipeline
# ─────────────────────────────────────────
async def run_filter_pipeline(
    file: UploadFile, content: bytes, extracted_text: str, word_count: int
) -> FilterResult:
    """
    Runs all 3 filter layers in sequence.
    Stops immediately on first rejection.
    """

    # Layer 1: Hard filter — no LLM, instant
    result = await hard_filter(file, content)
    if result:
        return result

    # Layer 2: Content classifier — Groq Llama 3.3
    result = await content_classifier(extracted_text, word_count)
    if result:
        return result

    # Layer 3: Safety check — Groq Llama 3.3
    result = await safety_filter(extracted_text)
    if result:
        return result

    # All passed — fetch topic for downstream use
    classification = await get_classification(extracted_text)

    return FilterResult(
        status=FilterStatus.APPROVED,
        message="Document passed all filters",
        word_count=word_count,
        detected_topic=classification.get("topic"),
        confidence=classification.get("confidence"),
    )
