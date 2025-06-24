FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PATH="/opt/venv/bin:$PATH"
CMD ["python", "LipCoordNet/api.py"]