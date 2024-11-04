FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p config cache

# Environment variables will be passed during runtime
ENV PYTHONPATH=/app

# Run scheduler test
CMD ["python", "-m", "src.scheduler_test"]