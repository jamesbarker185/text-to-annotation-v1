
# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set work directory
WORKDIR /app

# Install system dependencies (needed for OpenCV and others)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8095

# Command to run the application
CMD ["python", "main.py"]
