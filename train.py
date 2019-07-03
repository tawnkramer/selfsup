"""
train.py
Author: Tawn Kramer
Date: July 02 2019

Train a model with internally generated labels.
The first experiment is to localize sub patches.

The idea is to then take this trained network and seed your actual network for whatever
other task you need that will then require fewer examples.

> when using the filemask, we grab recursively. Works well if you use the syntax:
> --files=data/**/*.jpg

Usage:
    train.py [--files=<file_mask>] (--model=<model>)

Options:
    -h --help        Show this screen.
    -f --files=<file> File masks to grab files for training.
"""
import os
import numpy as np
import glob
import random

from docopt import docopt
import matplotlib.pyplot as plt
from tensorflow.python import keras

#training tasks
from tasks.localize_patch import LocalizePatch

def get_data_if_needed():
    """
    If no data dir, then go get some...
    """
    url = "https://tawn-train.s3.amazonaws.com/face_cars_dataset/data.zip"
    if not os.path.exists("data"):
        print('downloading large dataset (1GB)')
        os.system('wget %s' % url)
        os.system('unzip data.zip')
        os.unlink('data.zip')


def go(file_mask, model_path):
    """
    gather a bunch of images, pass them to our training,
    show a graph of results.
    """
    print("gathering files", file_mask)
    file_list = glob.glob(file_mask, recursive=True)    
    num_files = len(file_list)
    print("found %d files" % num_files)
    if num_files == 0:
        return

    #hyper-params
    batch_size = 64
    epochs = 10
    rate = 0.001
    decay = 0.0001
    train_perc = 0.8

    task = LocalizePatch()

    #shuffle list and split into two groups
    random.shuffle(file_list)
    split_line = round(train_perc * num_files)
    train_list = file_list[:split_line]
    val_list = file_list[split_line:]

    #calc how many steps per epoch
    steps_per_epoch = len(train_list) // batch_size
    val_steps = len(val_list) // batch_size
    
    print("len(train_list)", len(train_list))
    print("len(val_list)", len(val_list))
    print("steps_per_epoch", steps_per_epoch, "val_steps", val_steps)

    #these two generator functions serve up batches of data to the training
    train_gen, val_gen = task.get_generators(train_list, val_list, batch_size)

    #our model
    model = task.get_model()

    #our optimizer
    optimizer = keras.optimizers.Adam(lr=rate, decay=decay)

    #finalize before training
    task.compile_model(model, optimizer)

    #and train...
    history = model.fit_generator(train_gen, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=epochs,
                    verbose=1)

    model.save(model_path)

    plt.figure(1)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper right')
    plt.savefig(model_path + '_loss.png')
    plt.show()


if __name__ == "__main__":
    args = docopt(__doc__)    
    files = args['--files']
    model = args['--model']
    if files is None:
        files = "data/**/*.jpg"
    get_data_if_needed()
    go(files, model)
