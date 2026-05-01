import io
import fitz  # pymupdf
import docx


def extract_text(content: bytes, filename: str) -> tuple[str, int]:
    """
    Extracts text from PDF, DOCX, or TXT.
    Returns (text, word_count).
    """
    ext = filename.split(".")[-1].lower()

    if ext == "pdf":
        return _extract_pdf(content)
    elif ext == "docx":
        return _extract_docx(content)
    elif ext == "txt":
        return _extract_txt(content)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(content: bytes) -> tuple[str, int]:
    doc = fitz.open(stream=content, filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()
    text = "\n".join(pages_text).strip()
    return text, len(text.split())


def _extract_docx(content: bytes) -> tuple[str, int]:
    doc = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs).strip()
    return text, len(text.split())


def _extract_txt(content: bytes) -> tuple[str, int]:
    text = content.decode("utf-8", errors="ignore").strip()
    return text, len(text.split())
