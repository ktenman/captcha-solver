FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv and add to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    . /root/.local/bin/env

# Copy project files first for better caching
COPY pyproject.toml ./

# Use uv to compile dependencies and create requirements.lock
RUN uv pip compile pyproject.toml --format requirements -o requirements.lock
RUN uv pip install -r requirements.lock

# Copy model file and application code
COPY model.keras main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]