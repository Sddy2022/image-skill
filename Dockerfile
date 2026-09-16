# Dockerfile for image-skill
# GPU-enabled default using the official PyTorch CUDA runtime image.
# Build (GPU):
#   docker build -t image-skill:latest .
# Run (GPU):
#   docker run --gpus all -e HUGGINGFACE_TOKEN="<token>" -p 8000:8000 image-skill:latest
# CPU-only build (if you don't have GPU):
#   docker build --build-arg BASE_IMAGE=python:3.10-slim -t image-skill:cpu .
#   docker run -e HUGGINGFACE_TOKEN="<token>" -p 8000:8000 image-skill:cpu

ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-extra.txt ./

# Upgrade pip and install Python deps
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-extra.txt || true

# Note: some optional packages (xformers, realesrgan) may require manual installation steps
# If the pip install above fails for those, you can rebuild the image after adjusting requirements.

# Copy application code
COPY . /app

# Ensure non-root runtime user for safety
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

ENV OUTPUT_DIR=/app/outputs
RUN mkdir -p ${OUTPUT_DIR}

EXPOSE 8000

# Default command: start uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
