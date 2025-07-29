# Use official lightweight Python image as a parent image with platform specification
FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies required to build some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy only requirements separately to leverage Docker cache for dependencies
COPY requirements.txt .

# Install Python dependencies from requirements.txt with no cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence transformer model to avoid downloading at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-download nltk punkt tokenizer data for text processing
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy entire application code to /app directory in image
COPY . .

# Create directories for input and output files if they are needed by the app
RUN mkdir -p /app/input /app/output

# Default command to run the application
CMD ["python", "main.py"]
