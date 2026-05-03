# =============================================================================
# Dockerfile for Local Speech-to-Text API
# Uses Local Whisper Small + Sentence Transformers
# No OpenAI API key required - runs completely offline
# =============================================================================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install PyTorch CPU first (lighter version)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install numpy first (dependency for whisper)
RUN pip install numpy>=1.24.0

# Install openai-whisper
RUN pip install openai-whisper

# Install other Python dependencies
RUN pip install -r requirements.txt

# Download Whisper Medium model (~1.5GB) - just downloads, doesn't load into memory
RUN python -c "import whisper; whisper._download(whisper._MODELS['small'], '/root/.cache/whisper', False)"

# Download Sentence Transformers model (~500MB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy only the Python files from test2 folder
COPY local_api.py .
COPY local_whisper.py .
COPY local_analyzer.py .
COPY extract_local.py .
COPY arabic_corrector_local.py .
COPY audio_preprocessor_local.py .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port 8001
EXPOSE 8001

# Run the API
CMD ["uvicorn", "local_api:app", "--host", "0.0.0.0", "--port", "8001"]
