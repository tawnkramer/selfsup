from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, Dropout, SpatialDropout2D, Flatten

class BaseTask(object):

    def create_model(self, num_outputs, input_shape=(120, 160, 3), drop=0.2):
        """
        defined the NN layers for this model.
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

    def compile_model(self, model, optimizer):
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])