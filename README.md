# Pathology Tiger Algorithm (AIVIS)

Segmentation and Detection algorithm developped by TEAM:AIVIS

<img src="https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example/blob/main/Tiger%20-%20algorithm%20example.png" width="500" height="500">


## Introduction
The dockerfile is heavily borrowed from https://github.com/DIAGNijmegen/pathology-tiger-algorithm-example.


## Libs & Weight
===========libs=========: 
The libraries(mmsegmentation & mmdetection & yolov5) we used for segmentation and detection

===========Weight========:
To test the algorithm, you can used the pre-trained weight.
https://drive.google.com/drive/folders/12mKK-Ql4y6PDw8vKbPe6OSKle_N82sh0?usp=sharing

## Dockerfile
To run the algorithm, we recommend to build the docker image based on the Dockerfile provided.
#### export.sh:
The command lines used for build docker image.
Before building the docker image, you should put the downloaed weight to the folder libs/WEIGTH.

#### test.sh:
The command lines used for testing the algorithm.
Before testing the algorithm, you should change the path in the test.sh
(change the ..path/input to a dir which contains the image and the tissue background).
The segmentation and detection results can be found in the docker volumn refered tiger-output.

