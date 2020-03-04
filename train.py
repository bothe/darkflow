from config_options import options, base_data_dir
from darkflow.net.build import TFNet

options = options(base_data_dir)

print("You are training '{}' data with {} batch size for {} epochs and".format(base_data_dir.upper(),
                                                                               options["batch"],
                                                                               options["epoch"]))

# load network
tfnet = TFNet(options)

# start train
tfnet.train()
