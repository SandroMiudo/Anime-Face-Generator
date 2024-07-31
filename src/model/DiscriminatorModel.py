from keras import models
from keras import layers
from keras import initializers
from keras import metrics
from keras import regularizers
from keras import optimizers
from keras import losses
import tensorflow as tf
from keras import saving
from utility.models import ModelHelper

class DiscriminatorModel(models.Model):

    def __init__(self, i_shape:tuple[int,int,int], learning_rate):
        super().__init__()
        self._i_shape = i_shape
        self._bin_loss = losses.BinaryCrossentropy(label_smoothing=0.1)
        self._loc_optimizer = optimizers.Adam(learning_rate)
        self._loss_tracker = metrics.Mean(name="loss")
        self._conv_block_1 = ModelHelper.Conv2DBlockBuilder.construct(16, (3,3), 
            i_shape=self._i_shape)
        self._pool_layer_1 = layers.MaxPooling2D(pool_size=(2,2))
        self._drop_layer_1 = layers.Dropout(0.1)
        self._conv_block_2 = ModelHelper.Conv2DBlockBuilder.construct(32, (3,3))
        self._pool_layer_2 = layers.MaxPooling2D(pool_size=(2,2))
        self._drop_layer_2 = layers.Dropout(0.1)
        self._conv_block_3 = ModelHelper.Conv2DBlockBuilder.construct(64, (3,3))
        self._pool_layer_3 = layers.MaxPooling2D(pool_size=(2,2))
        self._drop_layer_3 = layers.Dropout(0.1)
        self._conv_block_4 = ModelHelper.Conv2DBlockBuilder.construct(128, (3,3))

        self._flatten_layer = layers.Flatten()
        self._dense_layer_1 = layers.Dense(8, activation='relu', 
            kernel_initializer='he_normal')
        self._dense_layer_2 = layers.Dense(1, activation='sigmoid',
            kernel_initializer='glorot_normal')

    def compute_loss(self, gen_images_output, real_images):
        real_images_true_value = tf.ones_like(real_images)
        fake_images_true_value = tf.zeros_like(gen_images_output)

        return self._bin_loss(real_images_true_value, real_images) + self._bin_loss(fake_images_true_value, gen_images_output)

    def call(self, inputs, training=None):
        x = ModelHelper.LayerIterator(self._conv_block_1)(inputs, training=training)
        x = self._pool_layer_1(x, training=training)
        x = self._drop_layer_1(x, training=training)
        x = ModelHelper.LayerIterator(self._conv_block_2)(x, training=training)
        x = self._pool_layer_2(x, training=training)
        x = self._drop_layer_2(x, training=training)
        x = ModelHelper.LayerIterator(self._conv_block_3)(x, training=training)
        x = self._pool_layer_3(x, training=training)
        x = self._drop_layer_3(x, training=training)
        x = ModelHelper.LayerIterator(self._conv_block_4)(x, training=training)
        x = self._flatten_layer(x, training=training)
        x = self._dense_layer_1(x, training=training)
        x = self._dense_layer_2(x, training=training)
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
        return self._loc_optimizer.learning_rate.numpy()

    def apply_grads(self, discrimi_grads):
        self._loc_optimizer.apply_gradients(zip(discrimi_grads, self.trainable_variables))

    def apply_summary(self, p_fn):
        self.summary(print_fn=p_fn)

    def get_compile_config(self):
        return {
            "model_optimizer" : saving.serialize_keras_object(self._loc_optimizer),
            "model_metric" : saving.serialize_keras_object(self._loss_tracker)
        }

    def compile_from_config(self, config):
        self._loc_optimizer = saving.deserialize_keras_object(config.pop("model_optimizer"))
        self._loss_tracker  = saving.deserialize_keras_object(config.pop("model_metric"))