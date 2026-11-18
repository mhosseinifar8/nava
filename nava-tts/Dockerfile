# Stage 1: Obtain or compile CrispASR (GGML/GGUF runtime for Chatterbox Persian GGUF TTS)
FROM debian:bookworm-slim AS crispasr-builder

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy

# Configure Liara mirrors for Debian apt
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i 's|http://deb.debian.org/debian-security|http://linux-mirror.liara.ir/repository/debian-security/|g' /etc/apt/sources.list.d/debian.sources && \
      sed -i 's|http://deb.debian.org/debian|http://linux-mirror.liara.ir/repository/debian/|g' /etc/apt/sources.list.d/debian.sources; \
    fi && \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://deb.debian.org/debian-security|http://linux-mirror.liara.ir/repository/debian-security/|g' /etc/apt/sources.list && \
      sed -i 's|http://deb.debian.org/debian|http://linux-mirror.liara.ir/repository/debian/|g' /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/crispasr-build
RUN mkdir -p /opt/crispasr/bin /opt/crispasr/lib && \
    ARCH="$(uname -m)" && \
    INSTALLED=0 && \
    if [ "$ARCH" = "x86_64" ]; then \
      echo "Attempting prebuilt CrispASR Linux x86_64 release download..." && \
      if curl -fsSL --retry 2 --connect-timeout 20 \
        "https://github.com/CrispStrobe/CrispASR/releases/latest/download/crispasr-linux-x86_64.tar.gz" \
        -o crispasr.tar.gz && tar -xzf crispasr.tar.gz; then \
        BIN_FOUND="$(find . -maxdepth 4 -type f -name 'crispasr' | head -n 1)" && \
        if [ -n "$BIN_FOUND" ]; then \
          cp "$BIN_FOUND" /opt/crispasr/bin/crispasr && \
          chmod +x /opt/crispasr/bin/crispasr && \
          find . -maxdepth 4 \( -type f -o -type l \) -name '*.so*' -exec cp -P {} /opt/crispasr/lib/ \; && \
          INSTALLED=1; \
        fi; \
      fi; \
    fi && \
    if [ "$INSTALLED" -eq 0 ]; then \
      echo "Building CrispASR from source via CMake..." && \
      git clone --depth 1 --recursive https://github.com/CrispStrobe/CrispASR.git src && \
      cmake -S src -B build -DCMAKE_BUILD_TYPE=Release && \
      cmake --build build --target crispasr -j"$(nproc)" && \
      BIN_FOUND="$(find build -type f -name 'crispasr' | head -n 1)" && \
      cp "$BIN_FOUND" /opt/crispasr/bin/crispasr && \
      chmod +x /opt/crispasr/bin/crispasr && \
      find build \( -type f -o -type l \) -name '*.so*' -exec cp -P {} /opt/crispasr/lib/ \; ; \
    fi && \
    ls -la /opt/crispasr/bin /opt/crispasr/lib

# Stage 2: Final Nava TTS Runtime
FROM python:3.11-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://package-mirror.liara.ir/repository/pypi/simple \
    PIP_TRUSTED_HOST=package-mirror.liara.ir \
    NAVA_CRISPASR_BIN=/usr/local/bin/crispasr

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

# System dependencies for audio synthesis:
# - libsndfile1: WAV/FLAC decoding
# - espeak-ng & espeak-ng-data: Matcha-TTS & Piper Persian/English phonemization
# - libgomp1: OpenMP runtime for CrispASR (Chatterbox GGUF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    espeak-ng \
    espeak-ng-data \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CrispASR binary & shared libraries from builder stage
COPY --from=crispasr-builder /opt/crispasr/bin/crispasr /usr/local/bin/crispasr
COPY --from=crispasr-builder /opt/crispasr/lib/ /usr/local/lib/
RUN chmod +x /usr/local/bin/crispasr && ldconfig

# Configure Liara mirror for pip
RUN pip config set global.index-url https://package-mirror.liara.ir/repository/pypi/simple && \
    pip config set global.trusted-host package-mirror.liara.ir

# Dependencies for Piper, Pocket-TTS, Matcha-TTS & Ava-82M service
COPY nava-tts/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

# Install Ava-82M (Kokoro Persian 24kHz) runtime package
RUN pip install --no-cache-dir "https://huggingface.co/xmanii/Ava-82M/resolve/main/ava_tts-0.2.0-py3-none-any.whl"

# Install Nava CLI & Model Downloader (nava-download)
COPY pyproject.toml README.md ./
COPY nava ./nava
COPY nava_fa_speech ./nava_fa_speech
RUN pip install --no-deps -e .

COPY nava-tts/app ./app
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--loop", "uvloop", "--ws", "websockets", "--ws-max-size", "32000000", "--no-access-log"]
