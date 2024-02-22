from models.model import Model
from tensorflow.python.keras import Sequential, layers
from tensorflow.python.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.python.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential()
        pass
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        pass