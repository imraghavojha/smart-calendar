FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /smart-calendar

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Create any missing directories
RUN mkdir -p models/data models/logs

# Add project root to Python path
ENV PYTHONPATH="${PYTHONPATH}:/smart-calendar"

CMD ["python", "test_run.py"]