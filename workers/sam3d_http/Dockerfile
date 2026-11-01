# SAM 3D Objects behind the HTTP wire format of src/video2sim/models/sam3d_http.py.
# Mirrors the Modal image in src/video2sim/models/sam3d_modal.py (CUDA 12.1, Python 3.11, the dev,
# p3d and inference extras in that order, then the hydra patch). UNVERIFIED as a Docker build:
# pytorch3d and gsplat compile CUDA code, so build on a machine with the GPU you will run on
# (docker build --build-arg TORCH_CUDA_ARCH_LIST=8.0 ...). The gated weights are not baked in:
# mount a directory with checkpoints/pipeline.yaml at /weights (see README.md).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG SAM3D_COMMIT=f91db411c50efee93d8db7aeb323885650f6f722
ARG TORCH_CUDA_ARCH_LIST=8.0
ENV DEBIAN_FRONTEND=noninteractive FORCE_CUDA=1 CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121" \
    PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libgl1 libglib2.0-0 python3.11 python3.11-dev python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/facebookresearch/sam-3d-objects /opt/sam3d \
    && cd /opt/sam3d && git checkout ${SAM3D_COMMIT}
WORKDIR /opt/sam3d
RUN python -m pip install setuptools wheel && python -m pip install --no-build-isolation nvidia-pyindex \
    && python -m pip install -e '.[dev]' \
    && python -m pip install -e '.[p3d]' \
    && python -m pip install -e '.[inference]' \
    && python patching/hydra \
    && python -m pip install fastapi 'uvicorn[standard]' python-multipart
COPY server.py /srv/server.py
WORKDIR /srv
ENV SAM3D_CHECKPOINT_DIR=/weights/checkpoints SAM3D_REPO_ID=facebook/sam-3d-objects
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
