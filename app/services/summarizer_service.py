import asyncio
import google.generativeai as genai
from app.core.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel(settings.PROCESSING_MODEL)

# الـ free tier بيسمح بـ 5 RPM بس
# 13 ثانية بين كل call = مش هتتجاوز الـ limit أبداً
RATE_LIMIT_DELAY = 13


def _chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


async def _call_gemini(prompt: str) -> str:
    """Async Gemini call مع rate limit delay."""
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()


# ─────────────────────────────────────────
# Summarizer
# ─────────────────────────────────────────
async def summarize(text: str, topic: str | None = None) -> dict:
    words = text.split()
    if len(words) < 3000:
        return await _summarize_short(text, topic)
    else:
        return await _summarize_long(text, topic)


async def _summarize_short(text: str, topic: str | None) -> dict:
    topic_hint = f"The document is about: {topic}." if topic else ""

    prompt = f"""You are an expert summarizer. {topic_hint}
Summarize the following document clearly and concisely.

Document:
\"\"\"
{text}
\"\"\"

Respond in this exact format:
SUMMARY:
[Write a clear 3-5 paragraph summary covering the main ideas]

KEY_POINTS:
- [key point 1]
- [key point 2]
- [key point 3]
- [key point 4]
- [key point 5]

Rules:
- Be concise and clear
- Preserve the most important information
- Key points must be specific, not vague
- Use the same language as the document"""

    raw = await _call_gemini(prompt)
    return _parse_summary_response(raw)


async def _summarize_long(text: str, topic: str | None) -> dict:
    chunks = _chunk_text(text, chunk_size=3000, overlap=200)
    topic_hint = f"The document is about: {topic}." if topic else ""

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize this section of a document. {topic_hint}
Section {i+1} of {len(chunks)}:
\"\"\"
{chunk}
\"\"\"
Write a concise summary of this section in 2-3 paragraphs. Preserve key facts and ideas."""

        summary = await _call_gemini(prompt)
        chunk_summaries.append(summary)

        # Rate limit delay بين الـ chunks
        if i < len(chunks) - 1:
            await asyncio.sleep(RATE_LIMIT_DELAY)

    merged = "\n\n".join([f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)])

    # delay قبل الـ merge call
    await asyncio.sleep(RATE_LIMIT_DELAY)

    merge_prompt = f"""You have summaries of different sections of a document. {topic_hint}
Combine them into one coherent final summary.

Section summaries:
\"\"\"
{merged}
\"\"\"

Respond in this exact format:
SUMMARY:
[Write a clear 3-5 paragraph summary covering the main ideas of the full document]

KEY_POINTS:
- [key point 1]
- [key point 2]
- [key point 3]
- [key point 4]
- [key point 5]

Rules:
- Combine ideas from all sections coherently
- Remove repetition
- Key points must be specific, not vague
- Use the same language as the original document"""

    raw = await _call_gemini(merge_prompt)
    return _parse_summary_response(raw)


def _parse_summary_response(raw: str) -> dict:
    summary = ""
    key_points = []
    try:
        if "SUMMARY:" in raw and "KEY_POINTS:" in raw:
            parts = raw.split("KEY_POINTS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            key_points = [
                line.lstrip("- •*").strip()
                for line in parts[1].strip().split("\n")
                if line.strip() and line.strip().startswith(("-", "•", "*"))
            ]
        else:
            summary = raw
    except Exception:
        summary = raw
    return {"summary": summary, "key_points": key_points}
