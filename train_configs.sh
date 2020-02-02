#!/bin/bash
echo "download rest images"
wget --header="Host: doc-04-1g-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,ar-EG;q=0.8,ar;q=0.7" --header="Referer: https://drive.google.com/drive/my-drive" --header="Cookie: AUTH_u2j0t6bv6dftjmt2lsba7r3rvpi7gdis_nonce=rcsi44k0n7l34" --header="Connection: keep-alive" "https://doc-04-1g-docs.googleusercontent.com/docs/securesc/3dqr5f7kd72k46kubklelqd888nobt2a/1l4nt6ah5e3i1vu65ao4fdnuk6uvask4/1580637600000/01449274755406512808/01449274755406512808/1Yi_V_prQD3zY-nvXIFUXj9bPNAp-Auvv?e=download&authuser=0&nonce=rcsi44k0n7l34&user=01449274755406512808&hash=mtvacocjbqaj5rierttoor2sk83i7c4v" -O "images-rest.tar.gz" -c
tar -zxvf images-rest.tar.gz -C images/ && rm -r images-rest.tar.gz
mkdir bin
cd bin/
echo "download pretrained tiny yolo voc weights"
wget 'https://pjreddie.com/media/files/yolov2.weights'
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
