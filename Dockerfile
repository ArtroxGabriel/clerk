FROM python:3.14-slim

# Install system dependencies (ffmpeg is required for audio extraction/normalization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory inside the container for the application installation
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install project dependencies including the CUDA extra libraries
RUN uv sync --frozen --no-install-project --no-dev --extra cuda

# Copy project source files
COPY src/ ./src/
COPY README.md ./

# Build/install the project with the CUDA extra
RUN uv sync --frozen --no-dev --extra cuda

# Expose the virtual environment bin folder in PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.14/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.14/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Default directory for mount-based execution
WORKDIR /workspace

# Run the CLI tool
ENTRYPOINT ["meeting-pipeline"]
