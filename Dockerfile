# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including CMake
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set PATH to include the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Run the application
CMD ["python", "LipCoordNet/api.py"]