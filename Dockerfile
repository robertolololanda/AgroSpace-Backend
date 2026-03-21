# AgroSpace SMN WRF 4km Backend API
# FastAPI Python microservice that extracts Argentine National Met Service WRF 4km data

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for netCDF4
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libnetcdf-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
