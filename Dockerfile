# RiceGuard Deep Learning Backend - Production Container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENVIRONMENT=production

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch and torchvision
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn[standard]>=0.32.0 \
    pydantic>=2.10.0 \
    python-multipart>=0.0.20 \
    pillow>=11.0.0 \
    opencv-python-headless>=4.10.0 \
    numpy>=1.26.0 \
    scipy>=1.14.0 \
    pyyaml>=6.0.2

# Copy necessary codebase and frozen research artifacts
COPY src/ /app/src/
COPY app/ /app/app/
COPY configs/ /app/configs/
COPY experiments/phase3b_localization_refinement/ /app/experiments/phase3b_localization_refinement/
COPY results/figures/ /app/results/figures/

# Expose API port
EXPOSE 8000

# Run FastAPI backend via Uvicorn
CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
