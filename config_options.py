import os

# Directory of the data to be trained
# (it should contain: custom-yolov2.cfg, annotations, images, logs, and ckpt)
base_data_dir = "data_valves"


def options(base_data_dir, train=True):
    # Check if already trained
    if os.path.exists(base_data_dir + "/ckpt/checkpoint"):
        load = -1
    else:
        load = "bin/yolov2.weights"
        if not os.path.exists(load):
            print("bin/yolov2.weights is not available, initializing weights from scratch!")
            load = 0
    option_dict = {"model": base_data_dir + "/custom-yolov2.cfg",
                   'load': load,
                   "batch": 8,
                   "epoch": 1000,
                   'momentum': 0.9,
                   "trainer": 'adam',
                   'summary': base_data_dir + "/logs/",
                   "backup": base_data_dir + "/ckpt/",
                   "binary": base_data_dir + "/bin/",
                   "gpu": 1.0,
                   "train": train,
                   "save": 1000,
                   "threshold": 0.5,
                   "imgdir": base_data_dir + "/sample_img/",
                   "annotation": base_data_dir + "/annotations/",
                   "dataset": base_data_dir + "/images/",
                   "labels": base_data_dir + "/labels.txt"}

    return option_dict
