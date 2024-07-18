from keras import models
from keras import layers
from keras import initializers
from keras import metrics
from keras import regularizers
from keras import optimizers
from keras import losses
import tensorflow as tf

class DiscriminatorModel(models.Model):

    def __init__(self, i_shape:tuple[int,int,int], learning_rate):
        super().__init__()
        self._i_shape = i_shape
        self._bin_loss = losses.BinaryCrossentropy()
        self._loc_optimizer = optimizers.Adam(learning_rate)
        self._loss_tracker = metrics.Mean(name="loss")  
        self._conv_layer_1 = layers.Conv2D(filters=16, kernel_size=(5,5), padding='same', 
            activation='relu', input_shape=i_shape)
        # have changed the pool size to (2,2) for first two max pooling layers
        self._pool_layer_1 = layers.MaxPooling2D(pool_size=(2,2))
        self._conv_layer_2 = layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', 
            activation='relu')
        self._pool_layer_2 = layers.MaxPooling2D(pool_size=(2,2))
        self._conv_layer_3 = layers.Conv2D(filters=64, kernel_size=(4,4), padding='same', 
            activation='relu')
        self._pool_layer_3 = layers.MaxPooling2D(pool_size=(2,2))
        self._conv_layer_4 = layers.Conv2D(filters=128, kernel_size=(4,4), padding='same', 
            activation='relu')
        self._pool_layer_4 = layers.MaxPooling2D(pool_size=(2,2))
        self._flatten_layer = layers.Flatten()
        self._dense_layer_1 = layers.Dense(1, activation='sigmoid')

    def compute_loss(self, gen_images_output, real_images):
        real_images_true_value = tf.ones_like(real_images)
        fake_images_true_value = tf.zeros_like(gen_images_output)

        return self._bin_loss(real_images_true_value, real_images) + self._bin_loss(fake_images_true_value, gen_images_output)

    def call(self, inputs, training=None):
        x = self._conv_layer_1(inputs, training=training)
        x = self._pool_layer_1(x, training=training)
        x = self._conv_layer_2(x, training=training)
        x = self._pool_layer_2(x, training=training)
        x = self._conv_layer_3(x, training=training)
        x = self._pool_layer_3(x, training=training)
        x = self._conv_layer_4(x, training=training)
        x = self._pool_layer_4(x, training=training)
        x = self._flatten_layer(x, training=training)
        x = self._dense_layer_1(x, training=training)
        return x

    def get_config(self):
        return {
            "i_shape" : self._i_shape
        }
    
    @property
    def shape(self):
        return self._i_shape
    
    @property
    def learning_rate(self):
        return self._loc_optimizer.learning_rate

    def apply_grads(self, discrimi_grads):
        self._loc_optimizer.apply_gradients(zip(discrimi_grads, self.trainable_variables))

    def apply_summary(self, p_fn):
        self.summary(print_fn=p_fn)