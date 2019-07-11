from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, Dropout, SpatialDropout2D, Flatten, Conv2DTranspose

class BaseTask(object):

    def create_model(self, num_outputs, input_shape=(120, 160, 3), drop=0.2):
        """
        define the NN layers for this model.
        """

        img_in = Input(shape=input_shape, name='img_in')
        x = img_in
        x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(32, (5,5), strides=(1,1), activation='relu', name="conv2d_2")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(64, (5,5), strides=(1,1), activation='relu', name="conv2d_3")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', name="conv2d_5")(x)
        x = SpatialDropout2D(drop)(x)
        
        x = Flatten(name='flattened')(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(drop)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(drop)(x)

        outputs = Dense(num_outputs, activation='softmax', name="outputs")(x) 
            
        model = Model(inputs=img_in, outputs=outputs)
        
        return model

    def create_fully_conv_model(self, input_shape=(120, 160, 3), drop=0.2):
        """
        defined the NN layers for this model.
        """

        img_in = Input(shape=input_shape, name='img_in')
        padding = "same"
        x = img_in
        x = Convolution2D(24, (5,5), strides=(1,1), activation='relu', padding=padding, name="conv2d_1")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(32, (5,5), strides=(1,1), activation='relu', padding=padding, name="conv2d_2")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(64, (5,5), strides=(1,1), activation='relu', padding=padding, name="conv2d_3")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', padding=padding, name="conv2d_4")(x)
        x = SpatialDropout2D(drop)(x)
        x = Convolution2D(64, (3,3), strides=(1,1), activation='relu', padding=padding, name="conv2d_5")(x)
        x = Convolution2D(3, (1,1), strides=(1,1), activation='relu', padding=padding, name="fcn_output")(x)
        
        outputs = x 
            
        model = Model(inputs=img_in, outputs=outputs)
        
        return model

    def create_autoencoder_model(self, input_shape, drop = 0.2):
        """
        tweaked to work with (64, 64, 3) input and output
        """
        img_in = Input(shape=input_shape, name='img_in')
        x = img_in
        x = Convolution2D(24, (5,5), strides=(2,2), activation='relu', name="conv2d_1")(x)
        x = Dropout(drop)(x)
        x = Convolution2D(32, (5,5), strides=(1,1), activation='relu', name="conv2d_2")(x)
        x = Dropout(drop)(x)
        x = Convolution2D(32, (5,5), strides=(1,1), activation='relu', name="conv2d_3")(x)
        x = Dropout(drop)(x)
        x = Convolution2D(32, (3,3), strides=(1,1), activation='relu', name="conv2d_4")(x)
        x = Dropout(drop)(x)
        x = Convolution2D(32, (3,3), strides=(2,2), activation='relu', name="conv2d_5")(x)
        x = Dropout(drop)(x)
        x = Convolution2D(64, (3,3), strides=(2,2), activation='relu', name="conv2d_6")(x)
        x = Dropout(drop)(x)
        x = Convolution2D(64, (3,3), strides=(2,2), activation='relu', name="conv2d_7")(x)
        x = Dropout(drop)(x)
        x = Convolution2D(32, (1,1), strides=(2,2), activation='relu', name="latent")(x)
        
        y = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same", name="deconv2d_1")(x)
        y = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same", name="deconv2d_2")(y)
        y = Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding="same", name="deconv2d_3")(y)
        y = Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding="same", name="deconv2d_4")(y)
        y = Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding="same", name="deconv2d_5")(y)
        y = Conv2DTranspose(filters=3, kernel_size=(3,3), strides=2,  padding="same", name="img_out")(y)
        
        outputs = [y]
            
        model = Model(inputs=[img_in], outputs=outputs)

        return model


    def compile_model(self, model, optimizer):
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])