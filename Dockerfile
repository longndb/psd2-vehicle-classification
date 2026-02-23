# Vehicle Classification API - Docker Image

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    WINDOW_SECONDS=30.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY models/kaggle/cnn/averaged_model/best_model.pth ./results/convolutional_neural_network/averaged_model/best_model.pth

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the application with Gunicorn + Uvicorn workers
CMD exec gunicorn api.main:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -
