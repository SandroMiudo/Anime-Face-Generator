from keras import models
from keras import layers
import tensorflow as tf



class GeneratorModel(models.Model):
    """
    input_shape : size of the noise vector -> defaults to 164
    """
    def __init__(self, input_shape:int=164, higher_images_resolution=True):
        super().__init__()
        self.noise_vector_shape = input_shape
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

    def compute_loss(self, gen_images_output):
        fake_images_true_value = tf.ones_like(gen_images_output)
        return self.loss(fake_images_true_value, gen_images_output)
    
    def get_noise_vector_shape(self):
        return self.noise_vector_shape

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.dense_layer_1(x, training)
        x = self.batch_normalize_layer_1(x, training)
        x = self.leaky_relu_1(x, training)
        x = self.reshape_layer_1(x, training)
        x = self.conv_layer_1(x, training)
        x = self.batch_normalize_layer_2(x, training)
        x = self.leaky_relu_2(x, training)
        x = self.conv_layer_2(x, training)
        x = self.batch_normalize_layer_3(x, training)
        x = self.leaky_relu_3(x, training)
        if(self.higher_images_resolution):
            x = self.conv_layer_3(x, training)
            x = self.batch_normalize_layer_4(x, training)
            x = self.leaky_relu_4(x, training)
            x = self.conv_layer_4(x, training)
        else:
            x = self.conv_layer3(x, training)
    
    def apply_grads(self, gen_grads):
        self.optimizer.apply_gradients(zip(gen_grads, self.trainable_variables))