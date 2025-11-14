FROM python:3.13-slim

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages: tesseract + OpenCV deps
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-guj \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port (for local clarity – Railway will still use $PORT)
EXPOSE 7860

# Start Streamlit using Railway's PORT environment variable
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
