# TrailerSplat pod image: everything cloud_train.py's bootstrap used to install on
# every pod, baked in once. Built by GitHub Actions (.github/workflows/build.yml)
# and pushed to ghcr.io. cloud_train.py checks /opt/trailersplat-image and falls
# back to installing everything itself if a pod ever boots without it.
#
# Same base as the bare pods (keeps RunPod's SSH + start.sh behaviour).
FROM runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive

# System packages. libegl1/libgl1: nerfstudio imports open3d, which needs them.
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ffmpeg git xvfb unzip bzip2 curl libegl1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# GPU COLMAP 4 (conda-forge CUDA build) with SIFT/matching on the GPU and the
# global mapper. libfaiss 1.9.0 + openimageio 3.1 are pinned because the colmap
# package forgets to declare them and newer faiss is ABI-incompatible
# (verified 2026-09-30). The wrapper scopes conda's libraries to colmap only so
# they never shadow torch's. The build machine has no GPU driver, so conda is told
# to assume CUDA 12.9 (CONDA_OVERRIDE_CUDA); the pods check colmap really runs.
RUN cd /root \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba \
 && CONDA_OVERRIDE_CUDA=12.9 MAMBA_ROOT_PREFIX=/root/mamba ./bin/micromamba create -y -p /opt/cm -c conda-forge \
      "colmap=4.0.4=*cuda*" "cuda-version=12.9" "libfaiss=1.9.0" "openimageio=3.1.*" \
 && rm -rf /root/mamba /root/bin \
 && printf '#!/bin/sh\nLD_LIBRARY_PATH=/opt/cm/lib exec /opt/cm/bin/colmap "$@"\n' > /usr/local/bin/colmap \
 && chmod +x /usr/local/bin/colmap \
 && test -x /opt/cm/bin/colmap \
 && (colmap -h | head -1 || echo "colmap -h needs a GPU driver here - checked on the pod instead")

# nerfstudio, pinned to the version verified 2026-07..09. Bump deliberately.
RUN python -m pip install --no-cache-dir --upgrade pip \
 && (pip install --no-cache-dir --ignore-installed blinker || true) \
 && pip install --no-cache-dir --upgrade-strategy only-if-needed nerfstudio==1.1.5 \
 && NS_EVAL="$(python -c 'import nerfstudio,os;print(os.path.join(os.path.dirname(nerfstudio.__file__),"utils","eval_utils.py"))')" \
 && sed -i 's/torch.load(load_path, map_location="cpu")/torch.load(load_path, map_location="cpu", weights_only=False)/' "$NS_EVAL" \
 && ns-train --help > /dev/null

# nvcc on PATH (gsplat JIT-compiles its CUDA kernels), and pre-compile those
# kernels now so pods skip the ~2-3 min compile. The build machine has no GPU,
# so the targets are listed explicitly - every card cloud_train.py rents:
# 8.6 A40/A5000/A6000/3090, 8.9 L4/L40S/4090 (+PTX for newer cards). Older cards:
# cloud_train.py deletes this cache so gsplat recompiles. MAX_JOBS=2: compiling
# with 4 parallel nvcc jobs ran the 16 GB build machine out of memory (exit 143).
# Non-fatal: if it fails, pods compile as before.
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"
RUN (MAX_JOBS=2 python -c "from gsplat.cuda._backend import _C; print('gsplat kernels:', _C)" \
     && ls -d /root/.cache/torch_extensions/*/gsplat_cuda) \
    || echo "!! gsplat pre-compile failed - pods will JIT-compile on first use"

# Marker cloud_train.py looks for to skip the install steps.
RUN echo "trailersplat-image v1 (nerfstudio 1.1.5, COLMAP 4.0.4 CUDA)" > /opt/trailersplat-image
