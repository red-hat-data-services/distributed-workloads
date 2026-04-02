ARG PYTHON_VERSION=312
ARG IMAGE_TAG=9.7-1764607356

FROM registry.access.redhat.com/ubi9/python-${PYTHON_VERSION}:${IMAGE_TAG}

ARG CONSTRAINTS_FILE=requirements_compiled.txt

LABEL name="ray-ubi9-py312-cpu" \
      summary="Python 3.12 image based on UBI9 for Ray (CPU only)" \
      description="CPU-only Python 3.12 image based on UBI9 for Ray" \
      io.k8s.display-name="Python 3.12 base image for Ray (CPU)" \
      io.k8s.description="CPU-only Python 3.12 image based on UBI9 for Ray" \
      authoritative-source-url="https://github.com/opendatahub-io/distributed-workloads"

USER 0
WORKDIR /opt/app-root/bin

# Update base image and install basic build tools
RUN yum upgrade -y && \
    yum install -y \
        openblas \
        openblas-devel \
        make \
        findutils \
        cmake \
    && yum clean all && \
    rm -rf /var/cache/yum/*

# Copy constraints file
COPY ${CONSTRAINTS_FILE} .

RUN pip install --no-cache-dir \
    -c ${CONSTRAINTS_FILE} \
    ray[default,data,train,tune,serve]==2.52.1 \
    sentencepiece==0.1.99 \
    transformers==4.34.0 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --extra-index-url https://wheels.developerfirst.ibm.com/ppc64le/linux/+simple/

# Configure Hugging Face cache directory and set proper permissions
# Required because the container runs as non-root (UID 1001),
# and transformers need write access to download/cache models
ENV HF_HOME=/opt/app-root/src/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/app-root/src/.cache/huggingface

RUN mkdir -p $HF_HOME && \
    chown -R 1001:0 /opt/app-root/src/.cache && \
    chmod -R g+rwX /opt/app-root/src/.cache

# Restore user workspace
USER 1001
WORKDIR /opt/app-root/src
