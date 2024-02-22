from models.model import Model
# from keras.layers.experimental.preprocessing import Rescaling
from keras import Sequential, layers
#from keras.optimizers import RMSprop, Adam
from keras.optimizers.rmsprop import RMSprop

import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        model = Sequential()
        filter_multiplier = 11
        # block 1
        model.add(Conv2D(1 * filter_multiplier, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(150, 150, 3)))
        model.add(MaxPooling2D((2, 2)))
        # block 2
        model.add(Conv2D(2 * filter_multiplier, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        # block 3
        model.add(Conv2D(3 * filter_multiplier, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(4 * filter_multiplier, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(5 * filter_multiplier, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(6 * filter_multiplier, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(7 * filter_multiplier, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))

        # don't add any more convolutional layers, the images can't handle it

        model.add(Flatten())

        model.add(Dense(7 * filter_multiplier, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(6 * filter_multiplier, activation='relu'))
        # model.add(Dense(5 * filter_multiplier, activation='relu'))
        # model.add(Dense(4 * filter_multiplier, activation='relu'))
        # model.add(Dense(3 * filter_multiplier, activation='relu'))

        model.add(Dense(3, activation='softmax'))

        # model.summary()
        self.model = model
        return model

        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001, momentum=0.9),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )