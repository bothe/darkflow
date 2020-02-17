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
           "epoch": 10000,
           'momentum': 0.9,
           "trainer": 'adam',
           'summary': base_data_dir + "/logs/",
           "backup": base_data_dir + "/ckpt/",
           "gpu": 1.0,
           "train": True,
           "save": 1000,
           "annotation": base_data_dir + "/annotations/",
           "dataset": base_data_dir + "/images/",
           "labels": base_data_dir + "/labels.txt"}

# load network
tfnet = TFNet(options)
tfnet.savepb()

import tensorflow as tf
from ext_graph_utils.freeze_graph import freeze_session
tf.set_learning_phase(0)

frozen_graph = freeze_session(tf.get_session(), output_names=[out.op.name for out in tfnet.out])
tf.train.write_graph(frozen_graph, "params/pb", "tf_model.pb", as_text=False)

# start train
tfnet.train()
