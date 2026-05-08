"""
Q&A Generator Service
---------------------
Generates question-and-answer pairs from document text.
Uses the LLM fallback chain (OpenRouter → Gemini → Groq).

For long documents: generates Q&A per chunk then picks the best ones.
"""
from app.services.llm_client import call_llm


def _chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    """Splits text into overlapping chunks."""
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
async def generate_qa(text: str, topic: str | None = None, num_questions: int = 5) -> list[dict]:
    """
    Generates Q&A pairs from document.
    - Short docs (< 3000 words): single pass
    - Long docs: generate per chunk, then pick best
    Returns list of {"question": ..., "answer": ...}
    """
    words = text.split()
    if len(words) < 3000:
        return await _qa_short(text, topic, num_questions)
    else:
        return await _qa_long(text, topic, num_questions)


async def _qa_short(text: str, topic: str | None, num_questions: int) -> list[dict]:
    topic_hint = f"The document is about: {topic}." if topic else ""

    prompt = f"""You are an expert educator. {topic_hint}
Generate exactly {num_questions} question-and-answer pairs from the document below.

Document:
\"\"\"
{text}
\"\"\"

Rules:
- Questions must test understanding and critical thinking, not just memorization
- Answers must be complete, accurate, and based only on the document
- Cover different parts and topics within the document
- Use the same language as the document
- Do NOT number the questions

Respond in this EXACT format, one pair at a time:
Q: [question here]
A: [answer here]
---"""

    raw = await call_llm(prompt)
    return _parse_qa(raw, num_questions)


async def _qa_long(text: str, topic: str | None, num_questions: int) -> list[dict]:
    """Generate Q&A per chunk, then select the best diverse ones."""
    chunks = _chunk_text(text, chunk_size=3000, overlap=200)
    topic_hint = f"The document is about: {topic}." if topic else ""
    per_chunk = max(2, num_questions // len(chunks) + 1)
    all_qa = []

    for i, chunk in enumerate(chunks):
        prompt = f"""You are an expert educator. {topic_hint}
Generate {per_chunk} question-and-answer pairs from this section.
Section {i+1} of {len(chunks)}:
\"\"\"
{chunk}
\"\"\"

Rules:
- Questions must test understanding, not memorization
- Answers must be complete and accurate
- Use the same language as the document
- Do NOT number the questions

Respond in this EXACT format:
Q: [question here]
A: [answer here]
---"""
        raw = await call_llm(prompt)
        all_qa.extend(_parse_qa(raw, per_chunk))

    # Pick best diverse questions if we have more than needed
    if len(all_qa) > num_questions:
        all_qa = await _pick_best_qa(all_qa, num_questions, topic)

    return all_qa[:num_questions]


async def _pick_best_qa(qa_list: list[dict], num_questions: int, topic: str | None) -> list[dict]:
    """Ask LLM to pick the most diverse and important questions."""
    topic_hint = f"Topic: {topic}." if topic else ""
    formatted = "\n".join([f"Q: {q['question']}\nA: {q['answer']}\n---" for q in qa_list])

    prompt = f"""You have these Q&A pairs from a document. {topic_hint}
Pick the best {num_questions} pairs that are most diverse and cover different aspects.

Q&A pairs:
\"\"\"
{formatted}
\"\"\"

Return ONLY the selected {num_questions} pairs in this EXACT format:
Q: [question here]
A: [answer here]
---"""

    raw = await call_llm(prompt)
    return _parse_qa(raw, num_questions)


# ─────────────────────────────────────────
# Parser
# ─────────────────────────────────────────
def _parse_qa(raw: str, expected: int) -> list[dict]:
    """Parses Q:/A: format into list of dicts."""
    qa_list = []
    blocks = raw.split("---")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        question = ""
        answer_lines = []
        in_answer = False

        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("Q:"):
                question = line[2:].strip()
                in_answer = False
            elif line.startswith("A:"):
                answer_lines = [line[2:].strip()]
                in_answer = True
            elif in_answer and line:
                answer_lines.append(line)

        answer = " ".join(answer_lines).strip()
        if question and answer:
            qa_list.append({"question": question, "answer": answer})

    return qa_list[:expected]
