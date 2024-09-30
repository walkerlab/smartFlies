FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-20.04
LABEL maintainer='tamagotchi_fixed_pkgs'
RUN apt-get update \
    && apt-get install -y libopenmpi-dev \
    && apt-get install -y tmux \
    && apt-get install -y htop 
RUN pip install --upgrade pip==24.0
RUN pip install scikit-learn==1.1.2 yapf==0.33.0 gym==0.21.0 h5py==3.11.0 spyder==6.0.0 \
stable_baselines3==1.7.0 moviepy==1.0.3 mpi4py==4.0.0 tqdm==4.66.5 urllib3==2.2.3 \
virtualenv==20.26.5 joblib==1.2.0 natsort==8.4.0 mpl_scatter_density setproctitle==1.3.3 \
statsmodels==0.14.1 "imageio==2.6.0" "imageio-ffmpeg==0.4.2" array2gif==1.0.4 datajoint==0.14.2 \
mlflow==2.16.2 psutil==5.9.5 pynvml==11.5.3 gymnasium==0.29.1
RUN pip install git+https://github.com/webermarcolivier/statannot.git
COPY ./orca.deb /root/orca.deb
RUN dpkg -i /root/orca.deb

# git cloRUN git clone https://github.com/example/example.git && cd example && git checkout 0123abcdef
