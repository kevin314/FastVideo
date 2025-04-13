# Use NVIDIA CUDA 12.4.1 with Ubuntu 20.04 as base image
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to path
ENV PATH /opt/conda/bin:$PATH

# Create conda environment
RUN conda create --name myenv python=3.10.0 -y

# Set up conda activation in shell scripts
SHELL ["/bin/bash", "-c"]

# Copy just the pyproject.toml first to leverage Docker cache
COPY pyproject.toml ./

# Create a dummy README to satisfy the installation
RUN echo "# Placeholder" > README.md

# Install the package in development mode with test dependencies
RUN conda init bash && \
    echo "conda activate myenv" >> ~/.bashrc && \
    source ~/.bashrc && \
    conda activate myenv && \
    pip install --no-cache-dir --upgrade pip && \
    #pip install --no-cache-dir wheel setuptools && \
    # Make sure we use CUDA-enabled PyTorch
    pip install --no-cache-dir -e .[dev] && \
    # Install flash-attention separately with no-build-isolation as noted in your comments
    pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation && \
    # Clean conda cache to reduce image size
    conda clean -afy

# Copy the rest of your code
COPY . .

# Create an entrypoint script that activates conda environment
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate myenv\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

CMD ["tail", "-f", "/dev/null"]