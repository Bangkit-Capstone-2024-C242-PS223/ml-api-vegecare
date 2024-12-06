FROM python:3.9-slim

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY app/ ./app/
COPY data/ ./data/
COPY models/ /app/models/
COPY run.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "run.py"]
