FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and related packages
RUN pip install \
    torch==2.1.2 \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric dependencies
RUN pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# Install torchdiffeq from GitHub
RUN pip install torchdiffeq

# Install torchcubicspline from GitHub
RUN pip install git+https://github.com/patrick-kidger/torchcubicspline.git

# Install other dependencies
RUN pip install \
    "numpy<2" \
    pandas \
    scipy \
    huggingface-hub \
    matplotlib \
    properscoring \
    xarray \
    wandb && \
    rm -rf /root/.cache/pip && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + || true

# Install ESMF build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    g++ \
    gcc \
    libopenmpi-dev \
    openmpi-bin \
    libnetcdf-dev \
    libnetcdff-dev \
    liblapack-dev \
    libblas-dev \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download and extract ESMF v8.9.0
RUN cd /tmp && \
    wget https://github.com/esmf-org/esmf/archive/refs/tags/v8.9.0.tar.gz && \
    tar -xzf v8.9.0.tar.gz && \
    rm v8.9.0.tar.gz

# Set ESMF environment variables
ENV ESMF_DIR=/tmp/esmf-8.9.0
ENV ESMF_INSTALL_PREFIX=/usr/local/esmf
ENV ESMF_COMPILER=gfortran
ENV ESMF_COMM=openmpi
ENV ESMF_NETCDF=nc-config
ENV ESMF_LAPACK=netlib
ENV ESMF_LAPACK_LIBS="-llapack -lblas"
ENV ESMF_BOPT=O

# Build and install ESMF
RUN cd ${ESMF_DIR} && \
    make
    
# Install ESMF and cleanup (separate layer)
RUN cd ${ESMF_DIR} && \
    make install && \
    rm -rf /tmp/esmf-8.9.0

# Set ESMFMKFILE environment variable for installed ESMF
ENV ESMFMKFILE=${ESMF_INSTALL_PREFIX}/lib/libO/Linux.gfortran.64.openmpi.default/esmf.mk
ENV PATH=${ESMF_INSTALL_PREFIX}/bin:${PATH}

# Install ESMPy from the ESMF source (NOT xesmf yet)
RUN cd /tmp && \
    wget https://github.com/esmf-org/esmf/archive/refs/tags/v8.9.0.tar.gz && \
    tar -xzf v8.9.0.tar.gz && \
    cd esmf-8.9.0/src/addon/esmpy && \
    python3 -m pip install . && \
    cd /tmp && \
    rm -rf esmf-8.9.0 v8.9.0.tar.gz && \
    rm -rf /root/.cache/pip

# install xesmf
RUN pip install git+https://github.com/pangeo-data/xesmf.git && \
    rm -rf /root/.cache/pip && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + || true

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]