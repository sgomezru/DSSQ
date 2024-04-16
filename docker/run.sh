#!/usr/bin/env bash

# Provide image name as first argument

docker run \
    -it \
    --net=host \
    --runtime=nvidia \
    --gpus all \
    --cpus=14 \
    --privileged \
    --ipc=host \
    --mount type=bind,source="/home/gomez/Data",target="/data/Data" \
    --mount type=bind,source="/home/gomez/DSSQ",target="/workspace/src" \
    $1
