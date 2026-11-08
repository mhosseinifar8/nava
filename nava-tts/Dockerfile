FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

# System dependencies for audio synthesis
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 git && rm -rf /var/lib/apt/lists/*

# Dependencies for Piper & Pocket-TTS service
COPY nava-tts/requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt requests

# Install Nava CLI & Model Downloader (nava-download)
COPY pyproject.toml README.md ./
COPY nava ./nava
COPY nava_fa_speech ./nava_fa_speech
RUN pip install --no-deps -e .

COPY nava-tts/app ./app
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--loop", "uvloop", "--ws", "websockets", "--ws-max-size", "32000000", "--no-access-log"]
