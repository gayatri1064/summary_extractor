# Use lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/usr/share/nltk_data

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create virtual environment and install Python packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download NLTK tokenizer (punkt) locally into the container
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt

# Download sentence transformer models locally into the container
RUN python download_minilm.py && python download_cross_minilm.py

# Set default command to run main.py
CMD ["python", "main.py"]
