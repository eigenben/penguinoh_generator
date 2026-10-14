FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 is the default on Ubuntu 24.04.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-venv \
    python-is-python3 \
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

# Install dependencies from the lockfile first so source changes don't invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

CMD ["penguinoh", "train"]
