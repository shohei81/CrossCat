FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- System tooling -------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        g++ \
        gfortran \
        git \
        libboost-all-dev \
        libatlas-base-dev \
        libbz2-dev \
        libeigen3-dev \
        libffi-dev \
        liblapack-dev \
        liblzma-dev \
        libopenblas-dev \
        libssl-dev \
        pkg-config \
        software-properties-common \
        python2.7 \
        python2.7-dev \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/pip/2.7/get-pip.py | python2.7
ENV PATH="/root/.local/bin:${PATH}"

# Keep pip/setuptools close to the versions that CrossCat expects (Py2-compatible).
RUN python2.7 -m pip install --upgrade \
        pip==20.3.4 \
        setuptools==44.1.1 \
        wheel==0.37.1

# Numerical stack and build helpers for probcomp/crosscat.
RUN python2.7 -m pip install \
        numpy==1.16.6 \
        pandas==0.24.2 \
        scipy==1.2.3 \
        cython==0.29.36 \
        pytest==4.6.11

ARG CROSSCAT_REPO=https://github.com/probcomp/crosscat.git
ARG CROSSCAT_REF=master
RUN git clone --single-branch --branch "${CROSSCAT_REF}" "${CROSSCAT_REPO}" /opt/probcomp/crosscat

WORKDIR /opt/probcomp/crosscat
RUN python2.7 -m pip install -e . && \
    ln -sfn src crosscat && \
    python2.7 setup.py build_py

ENV PYTHONPATH=/opt/probcomp/crosscat
WORKDIR /workspace
CMD ["bash"]
