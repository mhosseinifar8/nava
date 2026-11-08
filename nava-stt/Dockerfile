FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

# Dependencies for Faster-Whisper STT service
COPY nava-stt/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

# Install Nava CLI & Model Downloader (nava-download)
COPY pyproject.toml README.md ./
COPY nava ./nava
COPY nava_fa_speech ./nava_fa_speech
RUN pip install --no-deps -e .

COPY nava-stt/app ./app
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "uvloop", "--ws", "websockets", "--ws-max-size", "32000000", "--no-access-log"]
