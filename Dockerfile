FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed by newspaper3k and lxml
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    zlib1g-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Ensure pip is up to date, install all dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install newspaper3k html5lib && \
    python -c "import nltk"

# Download required NLTK resources
RUN python -m nltk.downloader punkt punkt_tab stopwords

COPY . .

EXPOSE 5000

CMD ["python", "server.py"]
