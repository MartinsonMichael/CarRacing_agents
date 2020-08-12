FROM nvidia/cuda:10.0-base-ubuntu18.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python3-pip \
       python3-dev \
       python-opengl \
       python3-setuptools

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

RUN apt-get update --fix-missing
RUN apt-get install -y git

RUN pip3 install git+git://github.com/denisyarats/dmc2gym.git
RUN pip3 install wandb
RUN apt-get -y install ffmpeg
RUN apt-get -y install cuda-libraries-dev-10-0
RUN pip install -q "cupy-cuda100 ${CUPY_VERSION}"

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# PUT ALL CHANGES UNDER THIS LINE

COPY asound.conf /etc/asound.conf

RUN mkdir /src
WORKDIR /src
