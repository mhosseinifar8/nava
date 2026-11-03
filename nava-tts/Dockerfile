FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 git && rm -rf /var/lib/apt/lists/* && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY app ./app
EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--loop", "uvloop", "--ws", "websockets", "--no-access-log"]
