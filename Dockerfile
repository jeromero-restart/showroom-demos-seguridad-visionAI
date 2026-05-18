FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support first (must precede requirements.txt to avoid
# pip picking up the CPU-only build from PyPI for torch==2.1.1)
RUN pip install --no-cache-dir \
    torch==2.1.1+cu121 \
    torchvision==0.16.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies (torch/torchvision already satisfied above)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download YOLOv8s weights into image (Approach A — avoids runtime download)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Create demo cameras directory and generate synthetic test videos using ffmpeg static build
# Using static ffmpeg binary avoids conflicts between NVIDIA CUDA repos and Ubuntu's ffmpeg
# dependencies (libsdl2, libavutil, etc.) which break apt install on GPU hosts.
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
         -o /tmp/ffmpeg.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /tmp && \
    mv /tmp/ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg && \
    rm -rf /tmp/ffmpeg.tar.xz /tmp/ffmpeg-*-amd64-static && \
    mkdir -p /app/demo_cameras && \
    ffmpeg -f lavfi -i color=c=blue:s=1280x720:d=45:r=30 \
           -f lavfi -i anullsrc=r=44100:cl=mono \
           -c:v libx264 -crf 23 -c:a aac -shortest \
           /app/demo_cameras/parking.mp4 && \
    ffmpeg -f lavfi -i color=c=green:s=1280x720:d=60:r=25 \
           -f lavfi -i anullsrc=r=44100:cl=mono \
           -c:v libx264 -crf 23 -c:a aac -shortest \
           /app/demo_cameras/street.mp4 && \
    ffmpeg -f lavfi -i color=c=red:s=1280x720:d=50:r=30 \
           -f lavfi -i anullsrc=r=44100:cl=mono \
           -c:v libx264 -crf 23 -c:a aac -shortest \
           /app/demo_cameras/building.mp4 && \
    rm /usr/local/bin/ffmpeg

# Application setup
WORKDIR /app
COPY . /app

# Create data directories with proper permissions
RUN mkdir -p /app/data /app/data/uploads /app/data/results && chmod -R 777 /app/data

EXPOSE 8000

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
