import tensorflow as tf

meta_path = '/home/bothe/darkproject/darkflow/data_valves/ckpt/custom-yolov2-21468.meta' # Your .meta file
output_node_names = ['output']    # Output nodes
path = '/home/bothe/darkproject/darkflow/data_valves/tflite_model/SavedModel'
cktp_path = '/home/bothe/darkproject/darkflow/data_valves/ckpt'


with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(cktp_path))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

#    builder = tf.saved_model.builder.SavedModelBuilder(cktp_path)
 #   builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], )
  #  builder.save()

   # tf.saved_model.simple_save(sess, path,
    #                           inputs={"input":["input"]},
     #                          outputs={"output": ["output"]})

    # Save the frozen graph
    with open('/home/bothe/darkproject/darkflow/data_valves/tflite_model/saved_model.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
    tf.train.write_graph(frozen_graph_def, './',
                         '/home/bothe/darkproject/darkflow/data_valves/tflite_model/good_frozen.pbtxt', as_text=True)


graph_file = "/home/bothe/darkproject/darkflow/data_valves/tflite_model"
in_n = ["input"]
out_n = ["output"]

converter = tf.lite.TFLiteConverter.from_saved_model(graph_file, in_n, out_n)
converter.post_training_quantize = True
tflite_model = converter.convert()

open("/home/bothe/darkproject/darkflow/data_valves/tflite_model/quantized_model.tflite", "wb").write(tflite_model)

