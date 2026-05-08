"""
Summarizer Service
------------------
Generates summary, key points, and a simplified explanation
of the document using the LLM fallback chain.

Outputs:
- summary    : Full academic-style summary
- key_points : 5 bullet points of the most important ideas
- explanation: Simple explanation as if teaching a beginner
"""
import asyncio
from app.services.llm_client import call_llm


def _chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """Splits long text into overlapping chunks to avoid token limits."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────
async def summarize(text: str, topic: str | None = None) -> dict:
    """
    Main summarize function.
    Runs summary + explanation in parallel for speed.
    Handles short and long documents differently.
    """
    words = text.split()

    if len(words) < 3000:
        # Run summary and explanation in parallel
        summary_task = asyncio.create_task(_summarize_short(text, topic))
        explanation_task = asyncio.create_task(_explain_simple(text, topic))
        summary_result, explanation = await asyncio.gather(summary_task, explanation_task)
    else:
        # Long docs: summary first (needs chunking), then explanation
        summary_result = await _summarize_long(text, topic)
        explanation = await _explain_simple(text[:8000], topic)  # first ~8000 chars enough for explanation

    return {
        "summary": summary_result["summary"],
        "key_points": summary_result["key_points"],
        "explanation": explanation,
    }


# ─────────────────────────────────────────
# Summary (Academic Style)
# ─────────────────────────────────────────
async def _summarize_short(text: str, topic: str | None) -> dict:
    topic_hint = f"The document is about: {topic}." if topic else ""

    prompt = f"""You are an expert academic summarizer. {topic_hint}
Summarize the following document clearly and concisely.

Document:
\"\"\"
{text}
\"\"\"

Respond in this EXACT format with no extra text before or after:
SUMMARY:
[Write a clear 3-5 paragraph summary covering the main ideas]

KEY_POINTS:
- [key point 1]
- [key point 2]
- [key point 3]
- [key point 4]
- [key point 5]

Rules:
- Be concise and informative
- Key points must be specific facts, not vague statements
- Use the same language as the document
- Do not add any text outside the format above"""

    raw = await call_llm(prompt)
    return _parse_summary_response(raw)


async def _summarize_long(text: str, topic: str | None) -> dict:
    """Map-reduce: summarize each chunk, then merge into final summary."""
    chunks = _chunk_text(text, chunk_size=3000, overlap=200)
    topic_hint = f"The document is about: {topic}." if topic else ""

    # Step 1: Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize this section of a document. {topic_hint}
Section {i+1} of {len(chunks)}:
\"\"\"
{chunk}
\"\"\"
Write a concise summary in 2-3 paragraphs. Preserve all key facts and ideas."""
        summary = await call_llm(prompt)
        chunk_summaries.append(summary)

    # Step 2: Merge all chunk summaries
    merged = "\n\n".join([f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)])

    merge_prompt = f"""You have summaries of different sections of a document. {topic_hint}
Combine them into one coherent final summary.

Section summaries:
\"\"\"
{merged}
\"\"\"

Respond in this EXACT format:
SUMMARY:
[Write a clear 3-5 paragraph summary of the full document]

KEY_POINTS:
- [key point 1]
- [key point 2]
- [key point 3]
- [key point 4]
- [key point 5]

Rules:
- Combine ideas coherently, remove repetition
- Key points must be specific, not vague
- Use the same language as the original document"""

    raw = await call_llm(merge_prompt)
    return _parse_summary_response(raw)


# ─────────────────────────────────────────
# Simple Explanation (Beginner Friendly)
# ─────────────────────────────────────────
async def _explain_simple(text: str, topic: str | None) -> str:
    """
    Generates a simple, beginner-friendly explanation of the document.
    Avoids technical jargon, uses analogies and simple language.
    """
    topic_hint = f"The document is about: {topic}." if topic else ""

    prompt = f"""You are a friendly teacher explaining a topic to someone with no background in it. {topic_hint}

Read this document and explain it in very simple terms:
\"\"\"
{text[:6000]}
\"\"\"

Write a simple, friendly explanation following these rules:
- Use simple everyday language — no technical jargon
- If you must use a technical term, immediately explain it in parentheses
- Use analogies and real-life examples to clarify complex ideas
- Write in short paragraphs (2-3 sentences each)
- Aim for 3-5 paragraphs total
- Make it feel like you're talking to a curious friend, not writing a textbook
- Use the same language as the document (Arabic if document is in Arabic, English if English)

Start directly with the explanation, no introduction like "Sure!" or "Of course!"."""

    return await call_llm(prompt)


# ─────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────
def _parse_summary_response(raw: str) -> dict:
    """Parses SUMMARY:/KEY_POINTS: format from LLM response."""
    summary = ""
    key_points = []
    try:
        if "SUMMARY:" in raw and "KEY_POINTS:" in raw:
            parts = raw.split("KEY_POINTS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            key_points = [
                line.lstrip("- •*").strip()
                for line in parts[1].strip().split("\n")
                if line.strip() and line.strip()[0] in ("-", "•", "*")
            ]
        else:
            # Fallback: treat entire response as summary
            summary = raw
            key_points = []
    except Exception:
        summary = raw
        key_points = []

    return {"summary": summary, "key_points": key_points}
