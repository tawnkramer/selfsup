#todo add a task to colorize a bw image
#todo add task to test for detection of the rotation of an image
import random

from PIL import Image
import numpy as np
from tensorflow.python.keras import backend as K 

from tasks.task import BaseTask


def iou(true, pred):

    intersection = true * pred

    notTrue = 1 - true
    union = true + (notTrue * pred)

    return K.sum(intersection)/K.sum(union)

class Colorize(BaseTask):

    def __init__(self):
        self.sample_shape = (64, 64, 3)
        self.img_shape = (64, 64, 3)
        self.num_cat = 4
        self.drop = 0.1

    def get_generators(self, train_list, val_list, batch_size):
        train_gen = self.generator(train_list, self.sample_shape, batch_size)
        val_gen = self.generator(val_list, self.sample_shape, batch_size)
        return train_gen, val_gen

    def get_model(self):
        model = self.create_fully_conv_model(input_shape=self.img_shape,
                drop=self.drop)
        return model

    def compile_model(self, model, optimizer):
        model.compile(optimizer=optimizer, loss="mean_squared_logarithmic_error", metrics=["acc"])

    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def three_ch_grey(self, grey):
        width, height = grey.shape

        grey3 = np.zeros([width, height, 3], dtype=np.dtype('float32'))
        grey3[...,0] = np.reshape(grey, (width, height))
        grey3[...,1] = np.reshape(grey, (width, height))
        grey3[...,2] = np.reshape(grey, (width, height))
        return grey3

    def generator(self, file_list, sample_shape, batch_size):
        """
        choose a central patch of sample_shape, then choose another patch in a random
        dir in 8 cardinal directions. Append the two patches together in a single image
        with the label of the dir choice.
        """
        list_size = len(file_list)
        iImage = 0
        dp = 50 # deviation_pixels
        cat_dir = [0, 90, 180, 270]
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
                #img_arr = img_arr / 255.0

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
                    label_patch = img_arr[cy - ph : cy + ph, cx - pw : cx + pw, ]

                    #convert to a 3 channel grey scale
                    try:
                        grey = self.rgb2gray(label_patch)
                        src_patch = self.three_ch_grey(grey)
                    except:
                        break
         
                    #check to make sure it's the right size
                    valid_patch = src_patch.shape == sample_shape

                    iTry += 1

                if valid_patch:
                    src_patch = src_patch / 255.0
                    label_patch = label_patch / 255.0

                    X.append(src_patch)
                    y.append(label_patch)

                else:
                    dropped = file_list.pop(iImage)
                    list_size = len(file_list)
                    #print('dropping', dropped)

                #on to the next image. shuffle the list as we loop around to start.
                iImage += 1
                if iImage >= list_size:
                    random.shuffle(file_list)
                    iImage = 0
            
            yield np.array(X), np.array(y)

