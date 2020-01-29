#!/bin/bash
mkdir bin
cd bin/
echo "download pretrained tiny yolo voc weights"
wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
cd ../
echo "downloading training data"
# images
wget http://download1646.mediafire.com/bi6g2z6w3rsg/2okuftipzimoemk/images.tar.gz
wget http://download1496.mediafire.com/9fqwysedd6sg/o0dsih49vgllcle/annotations.tar.gz
echo "extracting training data"
tar -xvzf images.tar.gz
tar -xvzf annotations.tar.gz
rm -r images.tar.gz
rm -r annotations.tar.gz
echo "making checkpoints directory"
mkdir ckpt
echo "installing darkflow"
pip install .
pip install -e .

echo "start training :)"
python train.py