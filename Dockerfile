# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image) and the image must
# have the git binary -- python:*-slim does not ship one, and without it the
# install below fails with "unable to detect version".
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN uv pip install --system --no-cache ".[server,cli]"

# Faces are mounted here; the image ships none.
ENV ARGUS_MASK_ASSETS_DIR=/assets
VOLUME /assets

EXPOSE 8109
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8109/health', timeout=4)"
# 0.0.0.0 inside the container; which host address it is published on is compose's choice.
CMD ["argus-mask", "serve", "--host", "0.0.0.0", "--port", "8109"]

