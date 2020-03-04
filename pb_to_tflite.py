import os

import tensorflow as tf

from config_options import options, base_data_dir
from darkflow.net.build import TFNet

options = options(base_data_dir, train=False)

# load network
tfnet = TFNet(options)
tfnet.savepb(base_data=base_data_dir)

graph_file = base_data_dir + "/built_graph/custom-yolov2.pb"
in_n = ["input"]
out_n = ["output"]

tflite_path = base_data_dir + "/tflite_model/"
if not os.path.exists(tflite_path):
    os.mkdir(tflite_path)

converter1 = tf.lite.TocoConverter.from_frozen_graph(graph_file, in_n, out_n)
converter1.post_training_quantize = False
tflite_non_quantized_model = converter1.convert()

open(tflite_path + "non_quantized_model.tflite", "wb").write(tflite_non_quantized_model)

converter2 = tf.lite.TocoConverter.from_frozen_graph(graph_file, in_n, out_n)
converter2.post_training_quantize = True
tflite_quantized_model = converter2.convert()

open(tflite_path + "quantized_model.tflite", "wb").write(tflite_quantized_model)
