FROM python:3.8.11-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage/allennlp

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/* 

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --default-timeout=1000
RUN pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --default-timeout=1000 
RUN spacy download en_core_web_sm
#RUN spacy download es_core_news_sm


# Copy code
COPY *.py ./
COPY *.txt ./
COPY prediction/ prediction/
COPY rst_file rst_file/
COPY config/ config/

ENTRYPOINT []
CMD ["/bin/bash"]
