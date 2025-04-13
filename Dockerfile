FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ca-certificates \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda create --name fastvideo-dev python=3.10.0 -y

SHELL ["/bin/bash", "-c"]

# Copy just the pyproject.toml first to leverage Docker cache
COPY pyproject.toml ./

# Create a dummy README to satisfy the installation
RUN echo "# Placeholder" > README.md

# Install the package in development mode with test dependencies
RUN conda run -n fastvideo-dev pip install --no-cache-dir --upgrade pip && \
    conda run -n fastvideo-dev pip install --no-cache-dir .[dev] && \
    conda run -n fastvideo-dev pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation && \
    conda clean -afy

COPY . .

EXPOSE 22