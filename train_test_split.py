import os
import shutil
import random
import sys


def train_test_split(ratio):
    images = os.listdir('images/')
    labels = os.listdir('c_annotations/')
    all_data = list(zip(sorted(images), sorted(labels)))
    random.shuffle(all_data)
    for element in all_data[:int(ratio*len(images))]:
        shutil.copy('images/'+element[0], 'train-images/')
        shutil.copy('c_annotations/'+element[1], 'train-labels/')
    for element in all_data[int(ratio*len(images)):]:
        shutil.copy('images/'+element[0], 'test-images/')
        shutil.copy('c_annotations/'+element[1], 'test-labels/')


if __name__ == "__main__":
    train_test_split(float(sys.argv[1]))
