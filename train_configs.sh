#!/bin/bash
echo "download rest images"
wget --header="Host: doc-04-1g-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,ar-EG;q=0.8,ar;q=0.7" --header="Referer: https://drive.google.com/drive/my-drive" --header="Cookie: AUTH_u2j0t6bv6dftjmt2lsba7r3rvpi7gdis_nonce=gc6f6usc9ik0k" --header="Connection: keep-alive" "https://doc-04-1g-docs.googleusercontent.com/docs/securesc/3dqr5f7kd72k46kubklelqd888nobt2a/haoidhco2l1o37g9sil3ug193i9n1u5m/1580810400000/01449274755406512808/01449274755406512808/1Yi_V_prQD3zY-nvXIFUXj9bPNAp-Auvv?e=download&authuser=0&nonce=gc6f6usc9ik0k&user=01449274755406512808&hash=f65uojbc6n89tiipnd64853pggs2ah6q" -O "images-rest.tar.gz" -c
tar -zxvf images-rest.tar.gz -C images/ && rm -r images-rest.tar.gz
mkdir bin
cd bin/
echo "download pretrained tiny yolo voc weights"
wget 'https://pjreddie.com/media/files/yolov2.weights'
cd ../
echo "making checkpoints directory"
mkdir ckpt
echo "prepare train test samples"
mkdir train-images
mkdir train-labels
mkdir test-images
mkdir test-labels
mkdir c_annotations
echo "extract annots"
tar -zxvf c_annotations.tar.gz -C c_annotations && rm -r c_annotations.tar.gz
echo "train test split"
python train_test_split.py 0.7
echo "installing darkflow"
pip install .
pip install -e .

echo "start training :) <3"
python train.py
