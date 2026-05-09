FROM python:3.12-slim

# ─────────────────────────────────────────
# System dependencies
# libmagic  → file type detection
# libGL     → required by OpenCV (PaddleOCR dependency)
# libglib   → required by OpenCV
# ─────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR models at build time
# This avoids slow first-request downloads in production
# Pre-download PaddleOCR models at build time
# This avoids slow first-request downloads in production
RUN python -c "from paddleocr import PaddleOCR; print('Downloading English model...'); PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False); print('Downloading Arabic model...'); PaddleOCR(use_angle_cls=True, lang='arabic', use_gpu=False, show_log=False); print('All models downloaded!')"

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
