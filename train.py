import os

from darkflow.net.build import TFNet

# Directory of the data to be trained (should contain: custom-yolov2.cfg, annotations, images, logs, and ckpt)
base_data_dir = "data_valves"

# Check if already trained
if os.path.exists(base_data_dir + "/ckpt/checkpoint"):
    load = -1
else:
    load = "bin/yolov2.weights"
    if not os.path.exists(load):
        print("bin/yolov2.weights is not available, initializing weights from scratch!")
        load = 0

options = {"model": base_data_dir + "/custom-yolov2.cfg",
           'load': load,
           "batch": 8,
           "epoch": 500,
           'momentum': 0.9,
           "trainer": 'adam',
           'summary': base_data_dir + "/logs/",
           "backup": base_data_dir + "/ckpt/",
           "gpu": 1.0,
           "train": True,
           "save": 100,
           "annotation": base_data_dir + "/annotations/",
           "dataset": base_data_dir + "/images/",
           "labels": base_data_dir + "/labels.txt"}

# load network
tfnet = TFNet(options)
# start train
tfnet.train()
