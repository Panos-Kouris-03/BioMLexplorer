FROM python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for scanpy/tables/igraph; keep it lean
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-serial-dev \
    libxml2 \
    libxslt1.1 \
    libffi-dev \
    libgomp1 \
    libstdc++6 \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# If you already have requirements.txt, copy it; otherwise use the pinned block below
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -U pip wheel setuptools && \
    pip install -r requirements.txt

# bring your app code (and .streamlit/ if present)
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "mainapp.py", "--server.port=8501", "--server.address=0.0.0.0"]
LABEL authors="Panagiotis Kouris"

