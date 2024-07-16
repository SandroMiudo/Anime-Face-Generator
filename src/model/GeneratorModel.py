from keras import models
from keras import layers
import tensorflow as tf

class GeneratorModel(models.Model):
    """
    input_shape : size of the noise vector -> defaults to 256
    """
    def __init__(self, i_shape:int=128, higher_images_resolution=True):
        super().__init__()
        self.noise_vector_shape = i_shape
        self.higher_images_resolution = higher_images_resolution
        self.dense_layer_1 = layers.Dense(8*8*i_shape, input_shape=i_shape)

        self.reshape_layer_1 = layers.Reshape((8, 8, i_shape))
        self.dropout_layer_1 = layers.Dropout(rate=0.3)

        self.conv_layer_1 = layers.Conv2DTranspose(256, (2, 2), strides=(2,2), padding='same')
        self.leaky_relu_1 = layers.LeakyReLU()
        self.batch_normalize_layer_1 = layers.BatchNormalization()

        self.conv_layer_1_5 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')
        self.leaky_relu_2 = layers.LeakyReLU()

        self.dropout_layer_2 = layers.Dropout(rate=0.3)

        self.conv_layer_2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.leaky_relu_3 = layers.LeakyReLU()
        self.batch_normalize_layer_2 = layers.BatchNormalization()

        if higher_images_resolution: # (128,128)
            self.conv_layer_3 = layers.Conv2DTranspose(32, (4, 4), strides=(4, 4), padding='same')
            self.leaky_relu_4 = layers.LeakyReLU()
            self.conv_layer_4 = layers.Conv2D(3, (4,4), strides=(2,2), padding='same', activation='sigmoid')
        else: # (64,64)
            self.conv_layer3 = layers.Conv2D(3, (4, 4), strides=(1,1), padding='same', activation='sigmoid')

    def compute_loss(self, gen_images_output):
        fake_images_true_value = tf.ones_like(gen_images_output)
        return self.loss(fake_images_true_value, gen_images_output)
    
    def get_noise_vector_shape(self):
        return self.noise_vector_shape

    def call(self, inputs, training=None):
        x = self.dense_layer_1(inputs, training=training)
        x = self.reshape_layer_1(x, training=training)
        x = self.dropout_layer_1(x, training=training)
        x = self.conv_layer_1(x, training=training)
        x = self.leaky_relu_1(x, training=training)
        x = self.batch_normalize_layer_1(x, training=training)
        x = self.conv_layer_1_5(x, training=training)
        x = self.leaky_relu_2(x, training=training)
        x = self.dropout_layer_2(x, training=training)
        x = self.conv_layer_2(x, training=training)
        x = self.leaky_relu_3(x, training=training)
        x = self.batch_normalize_layer_2(x, training=training)

        if(self.higher_images_resolution):
            x = self.conv_layer_3(x, training=training)
            x = self.leaky_relu_4(x, training=training)
            x = self.conv_layer_4(x, training=training)
        else:
            x = self.conv_layer3(x, training=training)

        return x
    
    def get_config(self):
        input_shape = self.noise_vector_shape
        higher_res  = self.higher_images_resolution

        return {
            "i_shape" : input_shape, 
            "higher_images_resolution" : higher_res
        }

    @classmethod
    def from_config(cls, config):
        input_shape       =  config.pop("i_shape")
        higher_images_res = config.pop("higher_images_resolution")
        return cls(input_shape, higher_images_res)

    def apply_grads(self, gen_grads):
        self.optimizer.apply_gradients(zip(gen_grads, self.trainable_variables))