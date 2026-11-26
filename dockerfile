FROM python:3.11-slim

WORKDIR /app

# System deps for GDAL/rasterio, rio-tiler, and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libpq-dev \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment
ENV GDAL_CONFIG=/usr/bin/gdal-config

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend .

# Create output directory
RUN mkdir -p /app/output

ENV PYTHONPATH=/app
ENV OUTPUT_DIR=/app/output

EXPOSE 8000

CMD ["python", "-m", "eo_brain.api.main"]