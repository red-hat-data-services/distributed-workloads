FROM quay.io/modh/ray:2.35.0-py311-rocm62

ENV PYTORCH_ROCM_ARCH=gfx90a;gfx941;gfx942

# Install PyTorch
RUN python3 -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2

# Install Flash Attention
RUN python3 -m pip install wheel ninja

RUN export TMP_DIR=$(mktemp -d) \
    && cd $TMP_DIR \
    && git clone --depth 1 --branch v2.6.3 https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && git submodule update --init \
    && GPU_ARCHS=$PYTORCH_ROCM_ARCH MAX_JOBS="16" python3 setup.py install --verbose \
    && rm -rf $TMP_DIR

# Install DeepSpeed
RUN DS_ACCELERATOR=cuda DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed==0.15.1 --global-option="build_ext" --global-option="-j8"
