import tensorflow as tf

graph_file = "built_graph/custom-yolov2.pb"
in_n = ["input"]
out_n = ["output"]

converter1 = tf.lite.TocoConverter.from_frozen_graph(graph_file, in_n, out_n)
converter1.post_training_quantize = False
tflight_model1 = converter1.convert()

open("tflite_model/non_quantized_model.tflite", "wb").write(tflight_model1)

converter2 = tf.lite.TocoConverter.from_frozen_graph(graph_file, in_n, out_n)
converter2.post_training_quantize = True
tflight_model2 = converter2.convert()

open("tflite_model/quantized_model.tflite", "wb").write(tflight_model2)
