echo "You are loading the configuration from config_options.py,
you can interrupt training and start generarting the model files once you get some checkpoints in 'ckpt' directory"

echo "start training :) <3"
python train.py

echo "generarting .pb graph model (in build_graph directory) and TFLITE models (in tflite_model directory):) <3"
python pb_to_tflite.py
