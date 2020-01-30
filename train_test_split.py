import os
import shutil
import random
def train_test_split(samples=200):
    images=os.listdir('images/')
    labels=os.listdir('annotations/')
    all_data=list(zip(sorted(images),sorted(labels)))
    random.shuffle(all_data)
    i=0
    for element in all_data:
        if i<samples:
            shutil.copy('images/'+element[0],'train-images/')
            shutil.copy('annotations/'+element[1],'train-labels/')
            i+=1
        else:
            return
if __name__ == "__main__":
    train_test_split()