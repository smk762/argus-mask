# The Audio2Face server for argus-mask: NVIDIA's Audio2Face-3D SDK (MIT), a small worker that
# keeps one model loaded, and a Python front that speaks mask's /drive protocol.
#
#   docker build -t argus-mask-a2f a2f/
#   docker run --gpus all -p 127.0.0.1:8110:8110 -v a2f-data:/data argus-mask-a2f
#
# Two stages. `build` compiles the SDK and the worker against the CUDA and TensorRT
# development packages; it is large (its apt layer unpacks to about 19 GB, a third of it
# static libraries) and is never shipped. `runtime` takes only the shared libraries the
# worker and trtexec load.
#
# No model is in the image. The server downloads one from Hugging Face on first start and
# builds its TensorRT engine on the host's own GPU; both are cached in the /data volume.

# ---- build ----------------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-base-ubuntu22.04 AS build

ARG TRT_VERSION=10.13.3.9-1+cuda12.9
ARG SDK_COMMIT=1ca0f02535ed774f5dbcd724a31cd486368dc783
ENV DEBIAN_FRONTEND=noninteractive

# Only the CUDA pieces the SDK links (cudart, cublas, curand, nvrtc) plus nvcc,
# instead of the full -devel image. TensorRT is pinned: the SDK wants >=10.13,<11.
RUN printf 'Package: libnvinfer* libnvonnxparsers*\nPin: version %s\nPin-Priority: 1001\n' "$TRT_VERSION" \
      > /etc/apt/preferences.d/tensorrt \
 && apt-get update && apt-get install -y --no-install-recommends \
      build-essential git git-lfs ca-certificates curl wget unzip python3 python-is-python3 \
      cuda-nvcc-12-9 cuda-cudart-dev-12-9 cuda-nvrtc-dev-12-9 libcublas-dev-12-9 libcurand-dev-12-9 \
      libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10 \
      libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev \
      libnvinfer-headers-dev libnvinfer-headers-plugin-dev libnvinfer-bin \
 && rm -rf /var/lib/apt/lists/*

# The SDK's scripts expect a TensorRT root with bin/ lib/ include/.
RUN mkdir -p /opt/tensorrt \
 && ln -s /usr/include/x86_64-linux-gnu /opt/tensorrt/include \
 && ln -s /usr/lib/x86_64-linux-gnu /opt/tensorrt/lib \
 && ln -s /usr/src/tensorrt/bin /opt/tensorrt/bin
ENV TENSORRT_ROOT_DIR=/opt/tensorrt CUDA_PATH=/usr/local/cuda \
    PATH=/usr/src/tensorrt/bin:/usr/local/cuda/bin:$PATH

RUN git clone https://github.com/NVIDIA/Audio2Face-3D-SDK.git /sdk \
 && cd /sdk && git checkout "$SDK_COMMIT" && git lfs pull
WORKDIR /sdk

# packman fetches cmake, ninja and the header/static deps (eigen, cnpy, AudioFile, ...).
RUN ./fetch_deps.sh release

# NVTX headers: the SDK is built with -DUSE_NVTX.
RUN apt-get update && apt-get install -y --no-install-recommends cuda-nvtx-12-9 \
 && rm -rf /var/lib/apt/lists/*

COPY worker/ /sdk/audio2face-sdk/source/samples/a2f-worker/
# Only the worker and the SDK library it links: not the SDK's tests, benchmarks or samples.
RUN echo 'add_subdirectory(a2f-worker)' >> audio2face-sdk/source/samples/CMakeLists.txt \
 && export PATH="$PATH:/sdk/_deps/build-deps/ninja" \
 && _deps/build-deps/cmake/bin/cmake -B _build/release -G Ninja -S . -DCMAKE_BUILD_TYPE=Release \
 && _deps/build-deps/cmake/bin/cmake --build _build/release --target a2f-worker --parallel

# ---- runtime --------------------------------------------------------------------------------
FROM nvidia/cuda:12.9.1-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# The front is one Python file with one dependency, pinned by hash.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
 && printf 'websockets==15.0.1 --hash=sha256:ac1e5c9054fe23226fb11e05a6e630837f074174c4c2f0fe442996112a6de4fb\n' \
      > /tmp/requirements.txt \
 && pip3 install --no-cache-dir --require-hashes --only-binary :all: -r /tmp/requirements.txt \
 && apt-get purge -y python3-pip && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/requirements.txt

# The shared libraries by their full versioned names (the patterns match the real files, not
# the symlinks beside them); ldconfig below recreates the links.
#   libnvinfer, libnvinfer_plugin        the worker runs the engine with these
#   libnvinfer_builder_resource (2 GB)   trtexec needs it to build the engine on first start
#   libnvonnxparser                      trtexec reads the ONNX model with it
#   libcublas, libcublasLt, libcurand    the SDK's own GPU maths
# libcudart is already in the base image.
COPY --from=build \
    /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.* \
    /usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource.so.10.* \
    /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.* \
    /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10.* \
    /usr/local/cuda-12.9/lib64/libcublas.so.12.* \
    /usr/local/cuda-12.9/lib64/libcublasLt.so.12.* \
    /usr/local/cuda-12.9/lib64/libcurand.so.10.* \
    /sdk/_build/release/audio2x-sdk/lib/libaudio2x.so \
    /opt/a2f/lib/
COPY --from=build /usr/src/tensorrt/bin/trtexec /sdk/_build/release/audio2face-sdk/bin/a2f-worker /opt/a2f/bin/
# The SDK is MIT: its notice travels with the binary built from it.
COPY --from=build /sdk/LICENSE.txt /opt/a2f/licenses/Audio2Face-3D-SDK-LICENSE.txt
COPY server.py /opt/a2f/server.py

# Not root. The volume takes this ownership the first time a named volume is mounted; a bind
# mount must be writable by uid 10001.
RUN echo /opt/a2f/lib > /etc/ld.so.conf.d/a2f.conf && ldconfig \
 && useradd --uid 10001 --create-home a2f && mkdir -p /data && chown a2f:a2f /data
USER a2f

# LD_LIBRARY_PATH as well as ldconfig: TensorRT opens libnvinfer_builder_resource by its file
# name, and the loader cache only knows libraries by their soname.
ENV PATH=/opt/a2f/bin:$PATH \
    LD_LIBRARY_PATH=/opt/a2f/lib:/usr/local/cuda/lib64 \
    PYTHONUNBUFFERED=1 \
    A2F_DATA=/data \
    A2F_TRT_VERSION=10.13.3.9
VOLUME /data

EXPOSE 8110
# Healthy means the server answers; whether the model is ready is in the reply's "ready".
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python3 -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('A2F_PORT', '8110'), timeout=4)"
CMD ["python3", "/opt/a2f/server.py"]
