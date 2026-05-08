"""
Document Extractor Service
--------------------------
Extracts text from PDF, DOCX, and TXT files.

PDF Strategy:
- PaddleOCR runs on EVERY page (Arabic + English + Handwriting)
- No conditional logic — OCR always used for maximum text extraction
- pymupdf only used to convert pages to images for OCR input
"""
import io
import fitz  # pymupdf — used only for page-to-image conversion
import docx
import numpy as np

# Lazy import — models loaded once on first use
_ocr_en = None
_ocr_ar = None


def _get_ocr_en():
    """Lazy-load English OCR model (loaded once, reused)."""
    global _ocr_en
    if _ocr_en is None:
        from paddleocr import PaddleOCR
        _ocr_en = PaddleOCR(
            use_angle_cls=True,   # handles rotated/upside-down text
            lang="en",
            use_gpu=False,
            show_log=False,
        )
    return _ocr_en


def _get_ocr_ar():
    """Lazy-load Arabic OCR model (loaded once, reused)."""
    global _ocr_ar
    if _ocr_ar is None:
        from paddleocr import PaddleOCR
        _ocr_ar = PaddleOCR(
            use_angle_cls=True,
            lang="arabic",
            use_gpu=False,
            show_log=False,
        )
    return _ocr_ar


def _is_arabic_text(text: str) -> bool:
    """Returns True if text contains significant Arabic characters."""
    if not text:
        return False
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars > len(text) * 0.2


def _page_to_image(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """Converts a PDF page to numpy array for OCR input."""
    import cv2
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _ocr_page(page_image: np.ndarray) -> str:
    """
    Runs BOTH Arabic and English OCR on a page image.
    Combines results — Arabic text + English text from same page.
    This handles:
    - Pure Arabic pages
    - Pure English pages
    - Mixed Arabic + English pages
    - Handwritten text (both languages)
    - Printed text
    """
    text_en = ""
    text_ar = ""

    # Run English OCR
    try:
        result_en = _get_ocr_en().ocr(page_image, cls=True)
        if result_en and result_en[0]:
            text_en = " ".join([
                line[1][0]
                for line in result_en[0]
                if line and line[1] and line[1][0].strip()
            ])
    except Exception as e:
        print(f"[OCR-EN] Failed: {e}")

    # Run Arabic OCR
    try:
        result_ar = _get_ocr_ar().ocr(page_image, cls=True)
        if result_ar and result_ar[0]:
            text_ar = " ".join([
                line[1][0]
                for line in result_ar[0]
                if line and line[1] and line[1][0].strip()
            ])
    except Exception as e:
        print(f"[OCR-AR] Failed: {e}")

    # Combine both:
    # If Arabic model found Arabic text and English model found English text
    # → keep both
    # If both found same text (Latin chars detected by Arabic model too)
    # → deduplicate by keeping the longer/more accurate one per language
    parts = []

    if _is_arabic_text(text_ar):
        parts.append(text_ar)  # Arabic content from Arabic model

    if text_en and not _is_arabic_text(text_en):
        parts.append(text_en)  # English content from English model

    # Fallback: if nothing detected above, use whichever has more text
    if not parts:
        parts.append(text_ar if len(text_ar) > len(text_en) else text_en)

    return "\n".join(parts).strip()


# ─────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────
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
        raise ValueError(f"Unsupported file type: .{ext}")


def _extract_pdf(content: bytes) -> tuple[str, int]:
    """
    Extracts text from ALL pages using PaddleOCR.
    Works on: scanned PDFs, image PDFs, text PDFs, handwritten PDFs.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    all_pages_text = []

    for page_num, page in enumerate(doc):
        try:
            page_img = _page_to_image(page)
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
