FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install requests
COPY app ./app
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "uvloop", "--ws", "websockets", "--ws-max-size", "32000000", "--no-access-log"]
