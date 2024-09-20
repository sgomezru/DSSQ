FROM nvcr.io/nvidia/pytorch:23.10-py3

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

### Utils
RUN apt-get install -y \
	htop \
	git \
	vim \
	binutils \
	tmux

### - PIP
RUN pip install \
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
    scikit-image \
	gdown

### - GIT
## setup
# move nvidia-stuff to separate folder
WORKDIR /workspace
RUN rm -r *
## Project Repo
WORKDIR /workspace/repositories
RUN git clone https://github.com/sgomezru/DSSQ.git
RUN mkdir -p /data/Data
RUN mkdir -p /src/DSSQ
RUN gdown --folder --id 17iOAe8IfiGL_oEs4x2hiQEpkFIjdwbJt -O /workspace/out/
ENV MOUNT_DATA_PATH="/data/Data"
ENV OUT_PATH="/workspace/out"

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
