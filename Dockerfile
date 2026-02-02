# Multi-stage Dockerfile for file organizer
FROM python:3.10-slim-bullseye AS base

# Install system dependencies
RUN apt-get update -o Acquire::Check-Valid-Until=false && apt-get install -y \
    libmagic1 \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 organizer

# ============================================
# Builder stage
# ============================================
FROM base AS builder

# Install build dependencies
RUN apt-get update -o Acquire::Check-Valid-Until=false && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# ============================================
# Final stage
# ============================================
FROM base

# Copy installed packages from builder
COPY --from=builder /root/.local /home/organizer/.local

# Set up working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /models /data/input /data/output /data/logs && \
    chown -R organizer:organizer /models /data /app

# Switch to non-root user
USER organizer

# Add local Python packages to PATH
ENV PATH="/home/organizer/.local/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Set environment variables for model caching
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV TORCH_HOME=/models

# Models will be downloaded to /models volume on first container run
# The volume persists across container restarts, so models only download once
# The download script automatically checks if models exist before downloading

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
