#!/usr/bin/env bash

# Define the help message
helpMessage=$(cat <<EOF
Usage: $0 [options]

This script builds the Docker image for the project.

Options:
  -h, --help      Display this help message and exit
  -t <name>       Process the name provided with the -t option
  -d <path>       Specify the data input path
  -o <path>       Specify the output path

Example:
  $0 -t projectA -d /path/to/data -o /path/to/output

EOF
)

# Initialize our variables
name=""
datapath=""
outpath=""

while getopts "hd:o:t:" opt; do
  case ${opt} in
    h )
      echo "$helpMessage"
      exit 0
      ;;
    t )
      name="$OPTARG"
      ;;
    d )
      datapath="$OPTARG"
      ;;
    o )
      outpath="$OPTARG"
      ;;
    \? )
      echo "Invalid Option: -$OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid Option: -$OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done

# Check if the required options are provided
if [ -z "$name" ]; then
    echo "Error: No name provided. Use -t option."
    exit 1
fi
if [ -z "$datapath" ]; then
    echo "Error: No data input path provided. Use -d option."
    exit 1
fi
if [ -z "$outpath" ]; then
    echo "Error: No data output path provided. Use -o option."
    exit 1
fi

# Docker run command
docker run \
  -it \
  --net=host \
  --runtime=nvidia \
  --gpus all \
  --cpus=14 \
  --privileged \
  --ipc=host \
  --mount type=bind,source="$outpath",target=/workspace/out \
  --mount type=bind,source="$datapath",target=/data/Data \
  "$name"
