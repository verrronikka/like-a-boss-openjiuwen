FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    TMPDIR=/tmp
WORKDIR /app
COPY requirements.txt ./
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements.lock
COPY run_agent.py harness.py ./
ENTRYPOINT ["python", "/app/run_agent.py"]
