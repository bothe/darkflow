#!/bin/bash
mkdir bin
cd bin/
echo "download pretrained tiny yolo voc weights"
wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
cd ../
echo "downloading training data"
# images
wget http://download1646.mediafire.com/ux1iy7ebzpgg/2okuftipzimoemk/images.tar.gz
wget http://download1496.mediafire.com/9fqwysedd6sg/o0dsih49vgllcle/annotations.tar.gz
echo "extracting training data"
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
rm -r images.tar.gz
rm -r annotations.tar.gz
echo "making checkpoints directory"
mkdir ckpt
echo "installing darkflow"
pip install .
pip install -e .

echo "start training :)"
python train.py
