import matplotlib.pyplot as plt
import numpy as np
from darkflow.net.build import TFNet
import cv2
options = {"model": "cfg/custom-yolov2.cfg", 
           "load": "bin/yolov2.weights",
           "batch": 8,
           "epoch": 500,
           'momentum': 0.7,
           "gpu": 1.0,
           "train": True,
           "annotation": "./train-labels/",
           "dataset": "./train-images/"}
# load network
tfnet = TFNet(options)
# start train
tfnet.train()
