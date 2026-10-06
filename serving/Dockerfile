# Use official Python 3.11 slim image for reproducible containerized ML serving
FROM python:3.11-slim

# Set environment variables for Python in container
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies required by LightGBM (libgomp1 for OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy runtime requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python runtime dependencies
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Copy application source code, E1 model artifacts, and Phase 8 monitoring package
COPY __init__.py .
COPY api/ ./api/
COPY hydration/ ./hydration/
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/
COPY src/ ./src/
COPY inference/ ./inference/
COPY monitoring/ ./monitoring/

# Enable both top-level and serving.* package imports inside container
RUN ln -s /app /app/serving

# Ensure Linux LF line endings for LightGBM model text parser compatibility
RUN sed -i 's/\r$//' models/E1/model.txt

# Expose FastAPI application port
EXPOSE 8000

# Start FastAPI model serving application with Uvicorn bound to 0.0.0.0:8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
