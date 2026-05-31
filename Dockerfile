# ── Stage: Base Python image ──────────────────────────────────────────────
# We use 'slim' variant: Debian with minimal packages installed
# This gives us a smaller final image (~less than full python:3.9)
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────
# gcc is needed to compile some Python packages (e.g., numpy extensions)
# We remove the apt cache after installing to keep the image small
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────
# Install PyTorch CPU-only version first to avoid downloading 1GB+ of NVIDIA CUDA packages
# which often fail with hash mismatch errors in Docker.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first (Docker layer caching optimization):
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the NLP model ────────────────────────────────────────────
# This downloads the model DURING the Docker build, not at runtime.
# Result: the model is baked into the image → fast container startup.
# The model (~420MB) is cached at ~/.cache/huggingface/
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-mpnet-base-v2'); \
               print('Model downloaded successfully')"

# ── Copy application code ─────────────────────────────────────────────────
# We copy code AFTER installing dependencies (layer caching):
# Changing code doesn't invalidate the pip install layer.
COPY . .

# ── Expose port ───────────────────────────────────────────────────────────
# Tell Docker which port the app uses (8000 is FastAPI's default)
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────
# Docker checks this every 30s. If it fails 3 times, container is "unhealthy".
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# ── Start command ─────────────────────────────────────────────────────────
# uvicorn: the ASGI server that runs FastAPI
# --host 0.0.0.0: listen on all interfaces (needed inside Docker)
# --port 7860: use port 7860
# --workers 2: handle 2 concurrent requests
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
