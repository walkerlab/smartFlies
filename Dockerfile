# FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-20.04
# Image Args
# ARG UBUNTU_VER
# to apptainer by apptainer build smartFlies_hyak_DATE.sif docker-daemon://smartflies-tamagotchi_dev_container_test:latest
ARG CUDA_VER=12.4.1

FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04
LABEL maintainer="Edgar Y. Walker <eywalker@uw.edu>, Daniel Sitonic <sitonic@uw.edu>"

# Deal with pesky Python 3 encoding issue
ENV LANG=C.UTF-8

# Prevent Debian/Ubuntu from asking questions
ENV DEBIAN_FRONTEND=noninteractive

# Install essential Ubuntu packages
# and upgrade pip
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y software-properties-common \
    git \
    wget \
    vim \
    curl \
    zip \
    unzip \
    fish \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


RUN JAX_CUDA_VER = ${CUDA_VER:0:2} | \
    pip3 install \
    numpy \
    scipy \ 
    scikit-learn \
    pandas \
    matplotlib \ 
    seaborn \
    jax[cuda$JAX_CUDA_VER_local] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \ 
    numpyro \
    pymc
    
WORKDIR /src

LABEL maintainer='tamagotchi_fixed_pkgs'
RUN apt-get update \
    && apt-get install -y libopenmpi-dev \
    && apt-get install -y tmux \
    && apt-get install -y htop \
    && apt-get install -y nvtop \
    && apt-get remove -y python3-blinker
RUN pip install --upgrade pip==24.0 setuptools==65.5.0
RUN pip install scikit-learn==1.1.2 yapf==0.33.0 h5py==3.11.0 spyder==6.0.0 \
moviepy==1.0.3 mpi4py==4.0.0 tqdm==4.66.5 urllib3==2.2.3 \
virtualenv==20.26.5 joblib==1.2.0 natsort==8.4.0 mpl_scatter_density==0.7 setproctitle==1.3.3 \
statsmodels==0.14.1 "imageio==2.6.0" "imageio-ffmpeg==0.4.2" array2gif==1.0.4 datajoint==0.14.2 \
    mlflow==2.16.2 psutil==5.9.5 pynvml==11.5.3  gym==0.21.0 stable_baselines3==1.7.0 gymnasium==0.29.1  \
    tensorboard etils importlib_resources tensorboard-plugin-profile tensorflow plotly
    
# RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/webermarcolivier/statannot.git
# COPY ./orca.deb /root/orca.deb
# RUN dpkg -i /root/orca.deb
# python3 -c "import jax; print(f'Jax backend: {jax.default_backend()}')"
# git cloRUN git clone https://github.com/example/example.git && cd example && git checkout 0123abcdef
