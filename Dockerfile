# Use official Python base image
FROM python:3.12-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install project dependencies using uv
RUN uv pip install . --system --torch-backend cpu
RUN uv pip install uvicorn==0.38.0 --system

# Install optional external dependencies
RUN uv pip install git+https://github.com/navchetna/tree-parser.git --system --torch-backend cpu

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "marker.scripts.server:app", "--host", "0.0.0.0", "--port", "8000"]
