# Use a lightweight Python base image (CPU example)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code, model, and data into the container
COPY train.py evaluate.py setB.pth ./
COPY data ./data

# Set environment (optional but nice)
ENV PYTHONUNBUFFERED=1
    
# By default, run evaluation
CMD ["python", "evaluate.py"]
