ARG INSTALL_CUDA=false

FROM python:3.14-slim AS builder

ARG INSTALL_CUDA

# Build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -fL https://github.com/yt-dlp/yt-dlp/releases/download/2026.07.04/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp \
    && rm -rf /var/lib/apt/lists/*

# Pin uv to a specific version (not :latest)
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_CUDA" = "true" ]; then \
    uv sync --frozen --no-install-project --no-dev --extra cuda; \
    else \
    uv sync --frozen --no-install-project --no-dev; \
    fi

# Copy source and install the project itself
COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_CUDA" = "true" ]; then \
    uv sync --frozen --no-dev --extra cuda && \
    find /app/.venv -name "*.a" -delete && \
    find /app/.venv -name "*.h" -delete; \
    else \
    uv sync --frozen --no-dev; \
    fi && \
    find /app/.venv -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# ---------- Stage 2: runtime ----------
FROM python:3.14-slim AS runtime

# ffmpeg + ca-certificates are the only runtime system deps needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd -g "$GID" app && \
    useradd -l -m -u "$UID" -g "$GID" app

WORKDIR /app

# Bring in only what's needed at runtime — no uv, no curl, no build cache
COPY --from=builder /usr/local/bin/yt-dlp /usr/local/bin/yt-dlp
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/README.md /app/README.md

ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.14/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.14/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# Default directory for mount-based execution
WORKDIR /workspace

USER app
LABEL org.opencontainers.image.source="https://github.com/ArtroxGabriel/local-meeting-pipeline"
LABEL org.opencontainers.image.description="Local meeting pipeline (faster-whisper + Ollama)"

ENTRYPOINT ["meeting-pipeline"]
