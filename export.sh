#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build --no-cache -t tigerexamplealgorithm_cuda_0624_v3 "$SCRIPTPATH"

SEGMENTATION_FILE="/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
DETECTION_FILE="/output/detected-lymphocytes.json"
TILS_SCORE_FILE="/output/til-score.json"

MEMORY=4g

echo "Creating volume..."
docker volume create tiger-output

echo "Running algorithm..."
docker run --rm \
        --memory=$MEMORY \
        --memory-swap=$MEMORY \
        --gpus all \
        --network=none \
        --cap-drop=ALL \
        --security-opt="no-new-privileges" \
        --shm-size=128m \
        --pids-limit=256 \
        -v /media/mingfan/SSD1T/TIGER/input/:/input/ \
        -v /media/mingfan/SSD1T/TIGER/output:/output/ \
        --entrypoint python \
        tigerexamplealgorithm_cuda_0624_v3 \
        -mtigeralgorithmexample

sudo docker save tigerexamplealgorithm_cuda_0624_v3 | gzip -c > tigerexamplealgorithm_cuda_0624_v3.tar.xz