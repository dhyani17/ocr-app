FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Tesseract + libs for cv2)
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

# Set the ENTRYPOINT with python -c so we don't use bash
ENTRYPOINT ["python3", "-c", "import os; import subprocess; port=os.getenv('PORT','7860'); subprocess.run(['streamlit','run','app.py','--server.port='+port,'--server.address=0.0.0.0'])"]
