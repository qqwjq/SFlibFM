FROM ubuntu:14.04

RUN apt-get -y install software-properties-common
RUN apt-get update && apt-get upgrade -y --fix-missing

RUN apt-get install -y build-essential \
  curl \
  wget \
  openssl \
  libcurl4-openssl-dev \
  libpq-dev \
  libxml2-dev \
  nano

RUN apt-get update && apt-get install -y libxslt-dev

RUN apt-get update && apt-get install -y git

RUN apt-get update && apt-get install -y \
  python \
  python-pip \
  python-dev \
  libblas-dev \
  libsasl2-dev \
  postgresql-9.3

RUN apt-get install -y \
  python-numpy \
  python-scipy \
  python-patsy

RUN pip install pandas --upgrade

RUN pip install patsy

RUN pip install cython

RUN apt-get update && apt-get install -y subversion


RUN git clone https://f5d65e3ca6919c2484a24ee7037c82bc38d31fcd:x-oauth-basic@github.com/stitchfix/aa.git /code/aa && \
  pip install -e /code/aa


RUN cd /code && \
  git clone https://731f6aee036fd5be9360703283f884721ccd4b22@github.com/stitchfix/diamond.git && \
  cd diamond && \
  sudo pip install -e .

RUN cd /code && \
  git clone https://f64fbd1c20726535ad1852a339f112b83937de82@github.com/stitchfix/SFlibFM.git && \
  cd SFlibFM && \
  python setup.py install

RUN cd /code && \
  git clone https://9cd562afc2efff1e673bda94b0233857ed202e12@github.com/stitchfix/svdplusplus.git && \
  cd svdplusplus && \
  python setup.py install
