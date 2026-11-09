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

# Install essential runtime and audio system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure Liara mirror for pip
RUN pip config set global.index-url https://package-mirror.liara.ir/repository/pypi/simple && \
    pip config set global.trusted-host package-mirror.liara.ir

# Copy project metadata and source code
COPY pyproject.toml README.md ./
COPY nava ./nava
COPY nava_fa_speech ./nava_fa_speech

# Install dependencies and Nava package
RUN pip install --upgrade pip && \
    pip install -e .

EXPOSE 8000

ENV NAVA_MODELS_DIR=/models

VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "nava.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
