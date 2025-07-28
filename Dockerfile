FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch manually first
RUN pip install --no-cache-dir torch==2.1.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Then install rest of dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything including model
COPY . .

# Force sentence-transformers to load local model path
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HOME=/root/.cache/huggingface

CMD ["python", "main.py"]
