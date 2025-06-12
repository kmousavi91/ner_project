# Base image with Python and spaCy installed
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to use Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Download German spaCy model if needed (optional step if not baked in)
RUN python -m spacy download de_core_news_md

# Default command for FastAPI server
CMD ["uvicorn", "apifast:app", "--host", "0.0.0.0", "--port", "8000"]

