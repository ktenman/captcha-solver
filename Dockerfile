FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first for better caching
COPY pyproject.toml ./

# Install uv and use it to manage dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv venv create && \
    /root/.local/bin/uv pip compile -o requirements.lock pyproject.toml && \
    /root/.local/bin/uv pip install -r requirements.lock
# Copy model file and application code
COPY model.keras main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]