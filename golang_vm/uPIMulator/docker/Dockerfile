FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV UPMEM_SDK_VERSION=2025.1.0
ENV UPMEM_SDK_PLATFORM=ubuntu-20.04

RUN apt update
RUN apt install -y git
RUN apt install -y git-lfs
RUN apt install -y wget
RUN apt install -y cmake
RUN apt install -y ninja-build
RUN apt install -y pkg-config
RUN apt install -y python3
RUN apt install -y python3-pip

# UPMEM SDK
WORKDIR /root
RUN git clone https://github.com/kagandikmen/upmem-sdk.git
RUN cd /root/upmem-sdk && git lfs install && git lfs pull
RUN cd /root/upmem-sdk/${UPMEM_SDK_VERSION}/${UPMEM_SDK_PLATFORM} && \
    tar -zxvf upmem-${UPMEM_SDK_VERSION}-Linux-x86_64.tar.gz -C /root
# Keep compatibility with hardcoded paths in benchmark/sdk CMake files.
RUN ln -s /root/upmem-${UPMEM_SDK_VERSION}-Linux-x86_64 /root/upmem-2023.2.0-Linux-x86_64
RUN echo "source /root/upmem-${UPMEM_SDK_VERSION}-Linux-x86_64/upmem_env.sh" > /root/.bashrc

WORKDIR /root/uPIMulator
