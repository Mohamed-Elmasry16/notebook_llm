FROM python:3.12-slim

# Install libmagic system dependency
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
