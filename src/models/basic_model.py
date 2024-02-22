from models.model import Model
from tensorflow.python.keras import Sequential, layers
#from tensorflow.python.keras.layers.experimental.preprocessing import Rescaling
#from keras.preprocessing import image
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
#from tensorflow.python.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential()
        
        self.model.add(layers.Conv2D(32, 
                                     (3, 3), 
                                     activation='relu', 
                                     kernel_initializer='he_uniform',
                                     padding='same',
                                     input_shape=(150, 150, 3)))
        self.model.add(layers.MaxPool2D((2, 2)))
        #self.model.add(layers.Flatten())
        #self.model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        #self.model.add(layers.Dense(1, activation='sigmoid'))
        #self.model.add(layers.Softmax())

        self.model.add(layers.Conv2D(64, 
                                     (3, 3), 
                                     activation='relu', 
                                     kernel_initializer='he_uniform',
                                     padding='same'))
        self.model.add(layers.MaxPool2D((2, 2)))
        #self.model.add(layers.Flatten())
        #self.model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        #self.model.add(layers.Dense(1, activation='sigmoid'))
        #self.model.add(layers.Softmax())

        self.model.add(layers.Conv2D(128, 
                                     (3, 3), 
                                     activation='relu', 
                                     kernel_initializer='he_uniform',
                                     padding='same'))
        self.model.add(layers.MaxPool2D((2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(layers.Dense(3, activation='sigmoid'))
        self.model.add(layers.Softmax())
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )