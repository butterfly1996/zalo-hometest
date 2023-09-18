FROM tensorflow/tensorflow:2.8.2-gpu
ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ=Asia/Ho_Chi_Minh
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    libsm6 \
    libxext6 \
    libturbojpeg \
    ffmpeg \
    libssl-dev \
    liblua5.1-0-dev \
    zlib1g-dev \
    ca-certificates \
    gcc libc-dev \
    libffi-dev \
    wget \
    python3-dev \
    python3-pip \
    curl \
    python3-levenshtein \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR ${WORKING_DIR}
ENV PYTHONPATH=${WORKING_DIR}
COPY requirements.txt ${WORKING_DIR}/requirements.txt
COPY . ${WORKING_DIR}
RUN python3 -m pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["python3", "api.py"]