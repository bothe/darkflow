import matplotlib.pyplot as plt
import numpy as np
from darkflow.net.build import TFNet
import cv2
options = {"model": "cfg/custom-yolov2.cfg", 
           "load": "bin/yolov2.weights",
           "batch": 8,
           "epoch": 50,
           'momentum': 0.9,
           "gpu": 1.0,
           "train": True,
           "annotation": "./train-labels/",
           "dataset": "./train-images/"}
# load network
tfnet = TFNet(options)
# start train
tfnet.train()
