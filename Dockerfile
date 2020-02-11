FROM python:3.6.9

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
       python-opengl

RUN pip3 install --upgrade pip
COPY pip.packages /tmp/
RUN pip3 install --trusted-host pypi.python.org -r /tmp/pip.packages

# PUT ALL CHANGES UNDER THIS LINE

RUN mkdir /src
WORKDIR /src
ADD . ./

