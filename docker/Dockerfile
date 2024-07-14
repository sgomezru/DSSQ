FROM nvcr.io/nvidia/pytorch:24.03-py3

# ARG user
# ARG uid
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video,utility,compute

ENV DEBIAN_FRONTEND noninteractive

# Before any libraries are installed, make sure everyone knows about cuda-aware ompi
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

### Update
RUN apt-get -y update && apt-get -y upgrade

### OPENGL
# RUN apt-get update && apt-get install -y \
# 	libassimp-dev \ 
# 	libjpeg-dev \
# 	libpng-dev \
# 	libglu1-mesa-dev \
# 	freeglut3-dev \
# 	mesa-common-dev \
# 	libgl1-mesa-dev \
# 	libegl1-mesa-dev

# RUN apt-get update && apt-get install -y -qq --no-install-recommends \
#     libglvnd0 \
#     libglx0 \
#     libxext6 \
#     libx11-6 \
#     libglfw3 \
#     libglfw3-dev \
#     libpcl-dev \
#     libfreetype6-dev

### Utils
RUN apt-get install -y \
	htop \
	git \
	vim \
	binutils \
	tmux

### - PIP
RUN pip install --upgrade \ 
	wandb \
	deep-pipe \
	scikit-learn \
	torchmetrics \
	tqdm \
	omegaconf \
	seaborn \
	monai \
	hydra-core \
    batchgenerators \
    numpy \
    pandas \
    scipy \
    nibabel \
    matplotlib \
    scikit-image

### - GIT
## setup
# move nvidia-stuff to separate folder
WORKDIR /workspace
RUN rm -r *
## Project Repo
WORKDIR /workspace/repositories
RUN git clone https://github.com/sgomezru/DSSQ.git
RUN mkdir -p /data/Data
ENV MOUNT_DATA_PATH="/data/Data"
RUN mkdir -p /src/DSSQ
ENV MOUNT_SRC_PATH="/src/DSSQ"

### - CLEANUP
RUN apt-get autoremove -y && rm -rf /tmp/* /var/tmp/* && apt-get clean && rm -rf /var/lib/apt/lists/*

### - Make user and give rights for workdir and folders we might need access to.
#RUN mkdir /out
#RUN useradd -u ${uid} -ms /bin/bash ${user}
#RUN chown -R ${user} /workspace /out /data /results

### - Expose ports for various services
# Make SSH available
EXPOSE 22
EXPOSE 42421
# TensorBoard https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 6006
# IPython https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 8889
### - start interactive console on attach
ENV DEBIAN_FRONTEND teletype 
