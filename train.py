"""
train.py
Author: Tawn Kramer
Date: July 02 2019

Train a model with internally generated labels.
The first experiment is to take a central patch and then choose
one of eight possible cardinal dir patches around and immediately 
adjascent to the patch. Then label the example with the cardinal dir
and ask the network to learn where the second patch came from. This 
is designed to encourage spatial awareness and ability to correlate features.

The idea is to then take this trained network and seed your actual network for whatever
other task you need that will then require fewer examples.

> when using the filemask, we grab recursively. Works well if you use the syntax:
> --files=data/**/*.jpg

Usage:
    train.py (--files=<file_mask> ...) (--model=<model>)

Options:
    -h --help        Show this screen.
    -f --files=<file> File masks to grab files for training.
"""
import os
import numpy as np
import glob
import random

from docopt import docopt
from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, Dropout, Flatten
from PIL import Image
import matplotlib.pyplot as plt



def create_model(num_outputs, input_shape=(120, 160, 3), drop=0.2):
    '''
    defined the NN layers for this model.
    '''

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5,5), strides=(1,1), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5,5), strides=(1,1), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = Dense(num_outputs, activation='softmax', name="outputs")(x) 
        
    model = Model(inputs=img_in, outputs=outputs)
    
    return model


def generator(file_list, sample_shape, batch_size):
    '''
    choose a central patch of sample_shape, then choose another patch in a random
    dir in 8 cardinal directions. Append the two patches together in a single image
    with the label of the dir choice.
    '''
    list_size = len(file_list)
    iImage = 0
    dp = 10 # deviation_pixels
    cat_dir = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
    pw = sample_shape[1] // 2
    ph = sample_shape[0] // 2
    dx = pw
    dy = ph
    max_tries = 10
    
    #just loop forever yielding the next batch_size array of data/labels pairs
    while True:
        X = []
        y = []

        for _ in range(batch_size):
            filename = file_list[iImage % list_size]
            img = Image.open(filename)

            #create float np array and normalize
            img_arr = np.array(img).astype(np.float32)
            img_arr = img_arr / 255.0

            cntr_x = img.width // 2
            cntr_y = img.height // 2

            valid_patch = False
            iTry = 0

            #sometimes we end up with a second patch that's off the edge of the image
            #give a few tries and then ditch the image if fails.
            while not valid_patch and iTry < max_tries:

                #choose a point near the center, with some random offset.
                cy = cntr_y + random.randint(-dp, dp)
                cx = cntr_x + random.randint(-dp, dp)

                #grab a central patch
                cntr_patch = img_arr[cy - ph : cy + ph, cx - pw : cx + pw, ]

                #choose a random dir
                iCat = random.randint(0, 7)
                label = np.zeros(8)
                label[iCat] = 1.0
                direction = cat_dir[iCat]
                offset_y = dy * direction[1]
                offset_x = dx * direction[0]

                #get a patch center in the dir off offset, with some jitter
                cy = cy + offset_y * 2 + random.randint(-dp, dp)
                cx = cx + offset_x * 2 + random.randint(-dp, dp)

                #grab an offset patch
                patch = img_arr[cy - ph : cy + ph, cx - pw : cx + pw, ]

                #check to make sure it's the right size
                valid_patch = patch.shape == sample_shape

                iTry += 1

            if valid_patch:
                #append the two in a single side, by side image
                comb_patch = np.append(cntr_patch, patch, axis=1)
                
                X.append(comb_patch)
                y.append(label)

            else:
                file_list.pop(iImage)
                list_size = len(file_list)

            #on to the next image. shuffle the list as we loop around to start.
            iImage += 1
            if iImage >= list_size:
                random.shuffle(file_list)
                iImage = 0
        
        yield np.array(X), np.array(y)



def go(file_mask, model_path):
    '''
    gather a bunch of images, pass them to our training,
    show a graph of results.
    '''
    print("gathering files", file_mask)
    file_list = glob.glob(file_mask, recursive=True)    
    num_files = len(file_list)
    print("found %d files" % num_files)
    if num_files == 0:
        return

    #hyper-params
    batch_size = 128
    sample_shape = (32, 32, 3)
    img_shape = (32, 64, 3)
    num_cat = 8
    epochs = 100
    rate = 0.001
    decay = 0.0001
    drop = 0.3
    train_perc = 0.8

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
    train_gen = generator(train_list, sample_shape, batch_size)
    val_gen = generator(val_list, sample_shape, batch_size)

    #our model
    model = create_model(num_outputs=num_cat, input_shape=img_shape, drop=drop)

    #our optimizer
    optimizer = keras.optimizers.Adam(lr=rate, decay=decay)

    #finalize before training
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    #print(model.summary())

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
    go(files[0], model)
