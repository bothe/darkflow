#!/bin/bash
mkdir bin
cd bin/
echo "download pretrained tiny yolo voc weights"
wget 'https://pjreddie.com/media/files/yolov2-tiny-voc.weights'
cd ../
echo "making checkpoints directory"
mkdir ckpt
echo "prepare train samples"
mkdir train-images
mkdir train-labels
python train_test_split.py 800
echo "installing darkflow"
pip install .
pip install -e .

echo "start training :)"
python train.py
