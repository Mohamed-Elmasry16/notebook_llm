import json
import httpx
from groq import Groq
from fastapi import UploadFile
from app.core.config import settings
from app.core.schemas import FilterResult, FilterStatus, RejectionReason

groq_client = Groq(api_key=settings.GROQ_API_KEY)

MAGIC_BYTES = {
    "pdf": [b"%PDF"],
    "docx": [b"PK\x03\x04"],
    "txt": [],
}


# ─────────────────────────────────────────
# LLM Callers
# ─────────────────────────────────────────
def _call_groq(prompt: str, max_tokens: int = 200) -> str:
    """Primary: Groq Llama 3.3."""
    response = groq_client.chat.completions.create(
        model=settings.CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


async def _call_openrouter_model(prompt: str, model: str, max_tokens: int = 200) -> str:
    """Generic OpenRouter caller — works for any model on OpenRouter."""
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://notebook-llm.app",
        "X-Title": "Notebook LLM",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{settings.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError(f"OpenRouter model {model} returned empty response")
        return content.strip()


async def _call_filter_llm(prompt: str, max_tokens: int = 200) -> str:
    """
    Filter LLM with 3-level fallback:
    1. Groq Llama 3.3          — Primary
    2. OpenRouter Qwen3 480B   — Backup 1
    3. OpenRouter Z.ai GLM 4.5 Air — Backup 2 (same API key, different model)
    """
    providers = [
        ("Groq Llama 3.3",
            lambda: _call_groq(prompt, max_tokens)),
        ("OpenRouter Qwen3 480B",
            lambda: _call_openrouter_model(prompt, settings.OPENROUTER_MODEL, max_tokens)),
        ("OpenRouter Z.ai GLM 4.5 Air",
            lambda: _call_openrouter_model(prompt, "z-ai/glm-4.5-air:free", max_tokens)),
    ]

    last_error = None
    for name, call in providers:
        try:
            if name == "Groq Llama 3.3":
                result = call()      # sync
            else:
                result = await call()  # async
            print(f"[Filter] Success: {name}")
            return result
        except Exception as e:
            last_error = e
            reason = "rate limit" if ("429" in str(e) or "rate" in str(e).lower()) else "error"
            print(f"[Filter] {name} failed ({reason}) — trying next...")
            continue

    raise RuntimeError(f"All filter providers failed. Last error: {last_error}")


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _check_magic_bytes(content: bytes, ext: str) -> bool:
    signatures = MAGIC_BYTES.get(ext, [])
    if not signatures:
        return True
    return any(content.startswith(sig) for sig in signatures)


# ─────────────────────────────────────────
# Layer 1: Hard Filter
# ─────────────────────────────────────────
async def hard_filter(file: UploadFile, content: bytes) -> FilterResult | None:
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if ext not in settings.ALLOWED_EXTENSIONS:
        return FilterResult(
            status=FilterStatus.REJECTED,
            reason=RejectionReason.INVALID_TYPE,
            message=f"File type '.{ext}' not supported. Allowed: pdf, docx, txt",
        )

    if not _check_magic_bytes(content, ext):
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

    raw = await _call_filter_llm(prompt, max_tokens=200)
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

    raw = await _call_filter_llm(prompt, max_tokens=200)
    return _parse_json(raw)


# ─────────────────────────────────────────
# Layer 3: Safety Filter
# ─────────────────────────────────────────
async def safety_filter(text_preview: str) -> FilterResult | None:
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

    raw = await _call_filter_llm(prompt, max_tokens=100)
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
    result = await hard_filter(file, content)
    if result:
        return result

    result = await content_classifier(extracted_text, word_count)
    if result:
        return result

    result = await safety_filter(extracted_text)
    if result:
        return result

    classification = await get_classification(extracted_text)

    return FilterResult(
        status=FilterStatus.APPROVED,
        message="Document passed all filters",
        word_count=word_count,
        detected_topic=classification.get("topic"),
        confidence=classification.get("confidence"),
    )
