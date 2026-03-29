# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base.
# Works for both in-repo builds (with vendored OpenEnv sources) and
# standalone builds (openenv-core from PyPI). The `openenv build` CLI
# sets appropriate build args based on context.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is required for installing VCS dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=dhara_ai

# Copy environment code.
COPY . /app/env
WORKDIR /app/env

# Ensure uv is on PATH.
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Sync dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ------------------------------------------------------------------ runtime
FROM ${BASE_IMAGE}

WORKDIR /app

# Bring over the virtual env and sources.
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Enable the Gradio web interface (required by CLAUDE.md for the hackathon harness).
ENV ENABLE_WEB_INTERFACE=true

# Hugging Face Spaces default port.
ENV PORT=7860
EXPOSE 7860

# Health check -- matches the port the server listens on.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the FastAPI server.
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
