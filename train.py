from darkflow.net.build import TFNet

options = {"model": "cfg/custom-yolov2.cfg",
           # "load": "bin/yolov2.weights",
           'load': -1,
           "batch": 8,
           "epoch": 50,
           'momentum': 0.9,
           'summary': 'logs/',
           "gpu": 1.0,
           "train": True,
           "save": 100,
           "annotation": "annotations/",
           "dataset": "images/"}
# load network
tfnet = TFNet(options)
# start train
tfnet.train()
