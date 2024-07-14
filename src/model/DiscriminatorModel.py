from keras import models
from keras import layers
import tensorflow as tf

class DiscriminatorModel(models.Model):

    def __init__(self, i_shape:tuple[int,int,int]):
        super().__init__()
        self.conv_layer_1 = layers.Conv2D(filters=64, kernel_size=(5,5), strides=(2,2),
            input_shape=i_shape)
        self.i_shape = i_shape
        self.leaky_relu_1 = layers.LeakyReLU()
        self.dropout_layer_1 = layers.Dropout(rate=0.3) # rate at which input will be replaced by 0
        self.conv_layer_2 = layers.Conv2D(filters=128, kernel_size=(5,5), strides=(2,2)) 
        self.leaky_relu_2 = layers.LeakyReLU()
        self.dropout_layer_2 = layers.Dropout(rate=0.3)
        self.flatten_layer = layers.Flatten() # out = (batch, filters)
        self.dense_layer   = layers.Dense(1)

    def compute_loss(self, gen_images_output, real_images):
        real_images_true_value = tf.ones_like(real_images)
        fake_images_true_value = tf.zeros_like(gen_images_output)
        return self.loss(real_images_true_value, real_images) + self.loss(fake_images_true_value, gen_images_output)
    
    def call(self, inputs, training=None):
        x = self.conv_layer_1(inputs, training=training)
        x = self.leaky_relu_1(x, training=training)
        x = self.dropout_layer_1(x, training=training)
        x = self.conv_layer_2(x, training=training)
        x = self.leaky_relu_2(x, training=training)
        x = self.dropout_layer_2(x, training=training)
        x = self.flatten_layer(x, training=training)
        x = self.dense_layer(x, training=training)

        return x

    def get_config(self):
        return {
            "i_shape" : self.i_shape
        }

    @classmethod
    def from_config(cls, config):
        input_shape = config.pop("i_shape")
        return cls(input_shape)

    def apply_grads(self, discrimi_grads):
        self.optimizer.apply_gradients(zip(discrimi_grads, self.trainable_variables))
