FROM python:3.11.9-slim

# 1. Install system dependencies needed for building some python wheels 
# and for matplotlib's font/rendering support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set environment variables to prevent Python from writing .pyc files
# and to ensure matplotlib doesn't try to open a GUI window (Agg backend)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pip install --no-cache-dir -r requirements.txt

# 3. Create a non-root user for security
RUN useradd -m agent
USER agent

WORKDIR /home/agent/workspace