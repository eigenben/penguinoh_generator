FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 is the default on Ubuntu 24.04.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    python3 \
    python3-venv \
    python-is-python3 \
    libglib2.0-0 \
    libgl1 \
    libjpeg-turbo8 \
    libpng16-16 \
  && rm -rf /var/lib/apt/lists/*

# Install uv (no curl/bootstrap needed).
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

WORKDIR /app

# Keep the venv outside the source tree.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Avoid matplotlib trying to use an interactive backend in containers.
ENV MPLBACKEND=Agg

# Sync dependencies from lockfile for reproducible, offline-ish runtime.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

CMD ["uv", "run", "--no-sync", "python", "main.py"]
