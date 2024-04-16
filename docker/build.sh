#!/usr/bin/env bash

# Provide docker image tag as first (only) argument

docker build \
    --pull \
    --progress=plain \
    --ssh default \
    -t $1 \
    -f Dockerfile .
