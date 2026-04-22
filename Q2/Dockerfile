FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git unzip curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    gdown \
    scikit-learn \
    matplotlib \
    streamlit \
    pillow \
    numpy

COPY train.py .
COPY app.py .

EXPOSE 7860

CMD ["bash"]
