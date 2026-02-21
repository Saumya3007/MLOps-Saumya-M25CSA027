# ╔══════════════════════════════════════════════════════════════╗
# ║  Dockerfile  —  DEVELOPMENT IMAGE                           ║
# ║  Purpose : full environment for training + evaluation       ║
# ╚══════════════════════════════════════════════════════════════╝

FROM python:3.10

LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="Dev: fine-tune DistilBERT on GPU/CPU + evaluate"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl vim wget build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# src/__init__.py is inside src/ so this one COPY covers everything
COPY src/ ./src/

ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV WANDB_DIR=/app/results/wandb
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/.cache/huggingface /app/results

VOLUME ["/app/results", "/app/.cache"]

CMD ["python", "-m", "src.train"]
