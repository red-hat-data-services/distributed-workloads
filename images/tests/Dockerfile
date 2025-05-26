# Use the official Go image as a base image
FROM golang:1.23

ENV KUBECONFIG=/distributed-workloads/tests/.kube/config

# Set the working directory inside the container
WORKDIR /distributed-workloads

# Copy the go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download && \
    go install gotest.tools/gotestsum@latest

WORKDIR /distributed-workloads/tests

# Copy the source from the current directory to the working directory inside the container
COPY tests/ .

# Command to run the tests
ENTRYPOINT [ "gotestsum"]

# Configure images using environment variables
# Corresponds to quay.io/modh/fms-hf-tuning:v2.8.2
ENV FMS_HF_TUNING_IMAGE=quay.io/modh/fms-hf-tuning@sha256:1ad46fe1a23f41f190c49ec2549c64f484c88fe220888a7a5700dd857ca243cc
# Corresponds to quay.io/modh/training:py311-cuda121-torch241
ENV TEST_TRAINING_CUDA_PYTORCH_241_IMAGE=quay.io/modh/training@sha256:f64f7bba3f1020d39491ac84d40d362a52e4822bdc11a33cfff021178b7c4097
# Corresponds to quay.io/modh/training:py311-rocm62-torch241
ENV TEST_TRAINING_ROCM_PYTORCH_241_IMAGE=quay.io/modh/training@sha256:883739b576b485d79966d8c894fdd9ebebd226605a2abe8b33593ca67c87a394
# Corresponds to quay.io/modh/training:py311-cuda124-torch251
ENV TEST_TRAINING_CUDA_PYTORCH_251_IMAGE=quay.io/modh/training@sha256:1d0caea3e5d56ff7d672954b1ad511e661df9bdb364d56879961169a4ca8dae0
# Corresponds to quay.io/modh/training:py311-rocm62-torch251
ENV TEST_TRAINING_ROCM_PYTORCH_251_IMAGE=quay.io/modh/training@sha256:6cdae840fa029da33cccab620367e82404d24ddf67762eb4537a9bffe1af306d
# Corresponds to quay.io/modh/odh-generic-data-science-notebook:v3-20250519
ENV NOTEBOOK_IMAGE=quay.io/modh/odh-generic-data-science-notebook@sha256:1c585c849cc466fce6443f5c186fb9b642d21b5f3581515ada08bba2fbc3591f
