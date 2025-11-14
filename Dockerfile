FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-guj \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8080

# Run using Python (not shell) so env PORT becomes integer correctly
CMD ["python3", "-u", "-c", "\
import os, subprocess; \
port = os.getenv('PORT', '8080'); \
print('Running Streamlit on port=' + port); \
subprocess.run(['streamlit', 'run', 'app.py', '--server.port=' + port, '--server.address=0.0.0.0']) \
"]
