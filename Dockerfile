# syntax=docker/dockerfile:1
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# hatch-vcs derives the version from git history, so the build context must
# include .git (keep it out of .dockerignore for this image) and the image must
# have the git binary -- python:*-slim does not ship one, and without it the
# build below fails with "unable to detect version". Nothing tracked may be
# left out of the context either: git would see it deleted, call the tree dirty
# and version the image 0.2.1.dev0+g<sha>.d<date> on the v0.2.0 tag (#88).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# The release passes the tag (without the v) as VERSION, so the image reports
# it whatever the tree looks like; a local `docker build` leaves it empty and
# git decides. The variable is set only for building this wheel: hatch-vcs
# ignores the per-package SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ARGUS_MASK, and
# the generic one would also version any dependency built from source.
ARG VERSION=""
RUN if [ -n "$VERSION" ]; then export SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION"; fi \
    && uv build --wheel --out-dir /tmp/dist \
    && uv pip install --system --no-cache "$(ls /tmp/dist/argus_mask-*.whl)[server,cli]" \
    && rm -rf /tmp/dist

# Faces are mounted here; the image ships none.
ENV ARGUS_MASK_ASSETS_DIR=/assets
VOLUME /assets

EXPOSE 8109
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8109/health', timeout=4)"
# 0.0.0.0 inside the container; which host address it is published on is compose's choice.
CMD ["argus-mask", "serve", "--host", "0.0.0.0", "--port", "8109"]
