#!/bin/bash
mkdir bin
cd bin/
# download pretrained tiny yolo voc weights
wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
cd ../
# checkpoints directory
mkdir ckpt
# install darkflow
pip install .
pip install -e .
