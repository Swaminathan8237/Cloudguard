# CloudGuard-Water: Production-Ready Dockerfile for FastAPI Prediction Microservice
FROM python:3.10-slim

# Set environment variables for non-interactive execution and unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained model artifacts
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Expose microservice port
EXPOSE 8000

# Container healthcheck using the native /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
