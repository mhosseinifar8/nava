FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://package-mirror.liara.ir/repository/pypi/simple \
    PIP_TRUSTED_HOST=package-mirror.liara.ir

WORKDIR /srv

# Configure Liara mirrors for Debian apt
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i 's|http://deb.debian.org/debian-security|http://linux-mirror.liara.ir/repository/debian-security/|g' /etc/apt/sources.list.d/debian.sources && \
      sed -i 's|http://deb.debian.org/debian|http://linux-mirror.liara.ir/repository/debian/|g' /etc/apt/sources.list.d/debian.sources; \
    fi && \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://deb.debian.org/debian-security|http://linux-mirror.liara.ir/repository/debian-security/|g' /etc/apt/sources.list && \
      sed -i 's|http://deb.debian.org/debian|http://linux-mirror.liara.ir/repository/debian/|g' /etc/apt/sources.list; \
    fi

# System dependencies for audio synthesis
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 git && rm -rf /var/lib/apt/lists/*

# Configure Liara mirror for pip
RUN pip config set global.index-url https://package-mirror.liara.ir/repository/pypi/simple && \
    pip config set global.trusted-host package-mirror.liara.ir

# Dependencies for Piper & Pocket-TTS service
COPY nava-tts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

# Install Nava CLI & Model Downloader (nava-download)
COPY pyproject.toml README.md ./
COPY nava ./nava
COPY nava_fa_speech ./nava_fa_speech
RUN pip install --no-deps -e .

COPY nava-tts/app ./app
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--loop", "uvloop", "--ws", "websockets", "--ws-max-size", "32000000", "--no-access-log"]
