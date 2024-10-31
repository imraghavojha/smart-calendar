FROM python:3.9-slim

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories and empty env file
RUN mkdir -p config cache && touch .env

# Default command
CMD ["python", "-m", "src.llm_scheduler"]