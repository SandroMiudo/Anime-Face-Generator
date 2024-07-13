from keras import models
from keras import layers

class GeneratorModel(models.Model):
    def __init__(self, input_shape:tuple[int,int,int], higher_images_resolution=True):
        super().__init__()
        self.higher_images_resolution = higher_images_resolution
        self.input_layer = layers.Input(input_shape)
        self.dense_layer_1 = layers.Dense(8*8*256, use_bias=False)
        self.batch_normalize_layer_1 = layers.BatchNormalization()
        self.leaky_relu_1 = layers.LeakyReLU()

        self.reshape_layer_1 = layers.Reshape((8, 8, 256))
        assert self.output_shape == (None, 8, 8, 256) # Note: None is the batch size

        self.conv_layer_1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        assert self.output_shape == (None, 8, 8, 128)
        self.batch_normalize_layer_2 = layers.BatchNormalization()
        self.leaky_relu_2 = layers.LeakyReLU()

        self.conv_layer_2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        assert self.output_shape == (None, 16, 16, 64)
        self.batch_normalize_layer_3 = layers.BatchNormalization()
        self.leaky_relu_3 = layers.LeakyReLU()

        if higher_images_resolution: # (128,128)
            self.conv_layer_3 = layers.Conv2DTranspose(32, (5, 5), strides=(4, 4), padding='same', use_bias=False)
            assert self.output_shape == (None, 64, 64, 32)
            self.batch_normalize_layer_4 = layers.BatchNormalization()
            self.leaky_relu_4 = layers.LeakyReLU()
            self.conv_layer_4 = layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
            assert self.output_shape == (None, 128, 128 , 3)
        else: # (64,64)
            self.conv_layer3 = layers.Conv2DTranspose(3, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation='tanh')
            assert self.output_shape == (None, 64, 64, 3)