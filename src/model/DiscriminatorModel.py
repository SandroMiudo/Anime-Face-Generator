from keras import models
from keras import layers

class DiscriminatorModel(models.Model):
    
    def __init__(self, input_shape:tuple[int,int,int]):
        super().__init__()
        self.input_layer  = layers.Input(input_shape)
        self.conv_layer_1 = layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2))
        self.leaky_relu_1 = layers.LeakyReLU()
        self.dropout_layer_1 = layers.Dropout(rate=0.3) # rate at which input will be replaced by 0
        self.conv_layer_2 = layers.Conv2D(filters=128, kernel_size=(5,5), strides=(2,2)) 
        self.leaky_relu_2 = layers.LeakyReLU()
        self.dropout_layer_2 = layers.Dropout(rate=0.3)
        self.flatten_layer = layers.Flatten() # out = (batch, filters)
        self.dense_layer   = layers.Dense(1) 