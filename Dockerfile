FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
