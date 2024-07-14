#!/usr/bin/env bash

# Define the help message
helpMessage=$(cat <<EOF
Usage: $0 [options]

This script builds the Docker image for the project.

Options:
  -h, --help      Display this help message and exit
  -t <name>       Process the name provided with the -t option

Example:
  $0 -t projectA  Builds an image with name projectA

EOF
)

# Initialize our variables
name=""

while getopts ":ht:" opt; do
  case ${opt} in
    h )
      echo "$helpMessage"
      exit 0
      ;;
    t )
      name="$OPTARG"
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

# Check if the name is provided
if [ -z "$name" ]; then
    echo "No name provided. Use -t option."
else
    echo "Processing name: $name"
fi

docker run \
	-it \
	--net=host \
	--runtime=nvidia \
    --gpus all \
    --cpus=14 \
    --privileged \
	--ipc=host \
	--mount type=bind,source="/home/gomez/out",target=/workspace/out \
	--mount type=bind,source="/home/gomez/Data/",target=/data/Data \
	--mount type=bind,source="/home/gomez/DSSQ",target=/src/DSSQ \
	$name
