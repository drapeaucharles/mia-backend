# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    requests==2.31.0

# Copy miner script
COPY run_miner.py .

# Make script executable
RUN chmod +x run_miner.py

# Health check to verify GPU access
RUN python3 -c "import subprocess; subprocess.run(['nvidia-smi'], check=False)"

# Default environment variables (can be overridden)
ENV MIA_API_URL=https://mia-backend-production.up.railway.app
ENV MINER_NAME=gpu-miner-001
ENV POLL_INTERVAL=5

# Run the miner
CMD ["python3", "run_miner.py"]