#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

SEGMENTATION_FILE="/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
DETECTION_FILE="/output/detected-lymphocytes.json"
TILS_SCORE_FILE="/output/til-score.json"

MEMORY=16g

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
        --shm-size=256m \
        --pids-limit=256 \
        -v /home/user/TIGER_TEST/input/:/input/ \
        -v /home/user/TIGER_TEST/output/:/output/ \
        --entrypoint python \
        tigerexamplealgorithm_l_bs_submit \
        -mtigeralgorithmexample


# docker run --rm --memory=16g --memory-swap=16g --gpus all --network=none --cap-drop=ALL --security-opt="no-new-privileges" --shm-size=256m --pids-limit=256 -v /home/user/TIGER_TEST/input/:/input/ -v /home/user/TIGER_TEST/output/:/output/ --entrypoint python tigerexamplealgorithm_l_bs_submit -mtigeralgorithmexample

# docker save tigerexamplealgorithm_l_bs_submit > tigerexamplealgorithm_l_bs_submit.tar