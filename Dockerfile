FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories for models and training data
RUN mkdir -p models/saved_models training_data/images training_data/metadata

# Copy application code
COPY . .

# Set permissions after copying all files
RUN chmod -R 777 models training_data

# Set environment variables
ENV PORT=7860

# Command to run the application
CMD ["python", "app.py"]
