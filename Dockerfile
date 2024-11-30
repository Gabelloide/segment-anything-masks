FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set non-interactive frontend for APT
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl wget \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    gcc g++ make cmake \
    ffmpeg x264 x265 \
    git aria2 \
    nano \
    libgl1-mesa-glx libglib2.0-0 \
    libavcodec-extra \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.11 get-pip.py \
    && rm get-pip.py \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip and install required Python packages
RUN python3.11 -m pip install --upgrade pip setuptools wheel


RUN git clone https://github.com/facebookresearch/segment-anything.git

# Create venv and install dependencies for SegmentAnything
RUN python3.11 -m venv /python_venv \
    && . /python_venv/bin/activate \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install opencv-python pycocotools matplotlib onnxruntime onnx wheel ninja \
    && pip3 install git+https://github.com/facebookresearch/segment-anything.git \
    && deactivate

# VENV
ENV VIRTUAL_ENV=/python_venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# CUDA 
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install GroundingDINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git \
    && . /python_venv/bin/activate \
    && cd GroundingDINO \
    && python setup.py build_ext --inplace \
    && python setup.py install \
    && cd .. \
    && deactivate

# Download models
RUN mkdir models \
    && cd models \
    && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam_vit_h.pth \
    && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O sam_vit_l.pth \
    && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O sam_vit_b.pth \
    && wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O groundingdino_swint_ogc.pth

# Move config file
RUN mkdir -p /configs
COPY GroundingDINO_SwinT_OGC.py /configs/

# Copy main code
COPY inference_on_a_image.py /
COPY run.sh /
RUN chmod +x run.sh

# output dirs
RUN mkdir -p /output/images
RUN mkdir -p /output/masks

# Fix for libs (.so files)
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}\
:/usr/lib64/python3.11/site-packages/torch/lib\
:/usr/lib/python3.11/site-packages/nvidia/cuda_cupti/lib\
:/usr/lib/python3.11/site-packages/nvidia/cuda_runtime/lib\
:/usr/lib/python3.11/site-packages/nvidia/cudnn/lib\
:/usr/lib/python3.11/site-packages/nvidia/cufft/lib"

