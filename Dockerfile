FROM --platform=linux/amd64 python:3.11-bullseye

WORKDIR /app

# Install system dependencies
# ffmpeg is needed for animation generation
# cmake, build-essential, g++ are needed for building some python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    cmake \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install dg-commons without dependencies to avoid dg-commonroad-drivability-checker
RUN pip install --no-cache-dir --no-deps git+https://github.com/idsc-frazzoli/dg-commons.git@pdm4ar/master

COPY src/ src/

ENV PYTHONPATH=/app/src

CMD ["python", "src/main.py"]
