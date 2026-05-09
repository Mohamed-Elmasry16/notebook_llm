"""
Document Extractor Service
--------------------------
Extracts text from PDF, DOCX, and TXT files.

PDF Strategy:
- Tesseract OCR runs on EVERY page (Arabic + English + Handwriting)
- No conditional logic — OCR always used for maximum text extraction
- pymupdf only used to convert pages to images for OCR input
"""
import io
import fitz  # pymupdf — used only for page-to-image conversion
import docx
import numpy as np
import pytesseract
from PIL import Image


# Tesseract language codes:
#   eng  → English
#   ara  → Arabic
#   eng+ara → both in one pass (handles mixed pages natively)
_LANG_EN = "eng"
_LANG_AR = "ara"
_LANG_MIXED = "eng+ara"

# Tesseract config: OEM 1 = LSTM engine (best accuracy), PSM 3 = auto page segmentation
_TSS_CONFIG = "--oem 1 --psm 3"


def _is_arabic_text(text: str) -> bool:
    """Returns True if text contains significant Arabic characters."""
    if not text:
        return False
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars > len(text) * 0.2


def _page_to_pil(page: fitz.Page, dpi: int = 200) -> Image.Image:
    """Converts a PDF page to a PIL Image for Tesseract input."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _ocr_page(page_image: Image.Image) -> str:
    """
    Runs Tesseract OCR on a page image in two passes:
      1. eng+ara — single combined pass that handles mixed-language pages.
      2. ara only  — second pass to catch Arabic text that the joint model
                     may under-detect when English dominates the layout.

    Combining both results covers:
    - Pure Arabic pages
    - Pure English pages
    - Mixed Arabic + English pages
    - Handwritten text (both languages, accuracy depends on trained data)
    - Printed text
    """
    text_mixed = ""
    text_ar = ""

    # Pass 1 — combined English + Arabic
    try:
        text_mixed = pytesseract.image_to_string(
            page_image, lang=_LANG_MIXED, config=_TSS_CONFIG
        ).strip()
    except pytesseract.TesseractError as e:
        print(f"[OCR-MIXED] Failed: {e}")

    # Pass 2 — Arabic-only pass (catches RTL text missed by joint model)
    try:
        text_ar = pytesseract.image_to_string(
            page_image, lang=_LANG_AR, config=_TSS_CONFIG
        ).strip()
    except pytesseract.TesseractError as e:
        print(f"[OCR-AR] Failed: {e}")

    # Merge results:
    # - Use mixed-pass output as the primary source (covers English well)
    # - Append Arabic-only result if it contains Arabic chars not already present
    parts = []

    if text_mixed:
        parts.append(text_mixed)

    if _is_arabic_text(text_ar) and text_ar not in text_mixed:
        parts.append(text_ar)

    # Fallback: if both passes returned nothing, return whatever exists
    if not parts:
        parts.append(text_mixed or text_ar)

    return "\n".join(parts).strip()


# ─────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────
def extract_text(content: bytes, filename: str) -> tuple[str, int]:
    """
    Extracts text from PDF, DOCX, or TXT.
    Returns (text, word_count).
    """
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        return _extract_pdf(content)
    elif ext == "docx":
        return _extract_docx(content)
    elif ext == "txt":
        return _extract_txt(content)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def _extract_pdf(content: bytes) -> tuple[str, int]:
    """
    Extracts text from ALL pages using Tesseract OCR.
    Works on: scanned PDFs, image PDFs, text PDFs.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    all_pages_text = []

    for page_num, page in enumerate(doc):
        try:
            page_img = _page_to_pil(page)
            page_text = _ocr_page(page_img)
            if page_text:
                all_pages_text.append(page_text)
                print(f"[OCR] Page {page_num + 1} extracted {len(page_text)} chars")
        except Exception as e:
            print(f"[OCR] Page {page_num + 1} failed: {e}")

    doc.close()

    full_text = "\n\n".join(all_pages_text).strip()
    word_count = len(full_text.split())
    print(f"[Extractor] Total: {word_count} words from {len(all_pages_text)} pages")

    return full_text, word_count


def _extract_docx(content: bytes) -> tuple[str, int]:
    """Extracts text from DOCX files."""
    document = docx.Document(io.BytesIO(content))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs).strip()
    return text, len(text.split())


def _extract_txt(content: bytes) -> tuple[str, int]:
    """Extracts text from plain text files."""
    text = content.decode("utf-8", errors="ignore").strip()
    return text, len(text.split())
