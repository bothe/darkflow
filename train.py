import matplotlib.pyplot as plt
import numpy as np
from darkflow.net.build import TFNet
import cv2
options = {"model": "cfg/custom-yolov2-tiny-voc.cfg", 
           "load": "bin/yolov2-tiny-voc.weights",
           "batch": 64,
           "epoch": 100,
           "gpu": 1.0,
           "train": True,
           "annotation": "./train-labels/",
           "dataset": "./train-images/"}
# load network
tfnet = TFNet(options)
# start train
tfnet.train()