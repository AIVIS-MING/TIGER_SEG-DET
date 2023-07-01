#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build --no-cache -t tigerexamplealgorithm_l_bs_submit "$SCRIPTPATH"
