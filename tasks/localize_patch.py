"""
localize_patch.py
Author: Tawn
Date: July 3, 2019

Brief:
Train a model with internally generated labels.
Take a central sub-patch from an image and then choose
one of eight possible cardinal dir patches around and immediately 
adjascent to the patch. Then label the example with the cardinal dir
and ask the network to learn where the second patch came from. This 
is designed to encourage spatial awareness and ability to correlate features.

"""
import random

from PIL import Image
import numpy as np

from tasks.task import BaseTask

class LocalizePatch(BaseTask):

    def __init__(self):
        self.sample_shape = (32, 32, 3)
        self.img_shape = (32, 64, 3)
        self.num_cat = 8
        self.drop = 0.1

    def get_generators(self, train_list, val_list, batch_size):
        train_gen = self.generator(train_list, self.sample_shape, batch_size)
        val_gen = self.generator(val_list, self.sample_shape, batch_size)
        return train_gen, val_gen

    def get_model(self):
        model = self.create_model(num_outputs=self.num_cat,
                input_shape=self.img_shape,
                drop=self.drop)
        return model

    def generator(self, file_list, sample_shape, batch_size):
        """
        choose a central patch of sample_shape, then choose another patch in a random
        dir in 8 cardinal directions. Append the two patches together in a single image
        with the label of the dir choice.
        """
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

