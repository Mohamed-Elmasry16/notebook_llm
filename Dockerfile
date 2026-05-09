FROM python:3.12-slim

# ─────────────────────────────────────────
# System dependencies
# libmagic      → file type detection
# tesseract-ocr → OCR engine
# tesseract-ocr-eng → English language data
# tesseract-ocr-ara → Arabic language data
# ─────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
