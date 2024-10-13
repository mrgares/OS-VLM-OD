FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Set environment variables 
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Update and install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \  
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from deadsnakes PPA and necessary development tools
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev

# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

# Install MongoDB shell
RUN apt-get update && apt-get install -y wget gnupg lsb-release && \
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add - && \
    echo "deb [ arch=amd64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list && \
    apt-get update && apt-get install -y mongodb-org-shell
   
# Set the working directory
WORKDIR /workspace

# Install any project-specific dependencies here
COPY requirements.txt .
RUN pip install -r requirements.txt


