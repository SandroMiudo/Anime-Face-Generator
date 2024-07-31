from keras import models
from keras import layers
from keras import initializers
from keras import losses
from keras import metrics
from keras import regularizers
from keras import optimizers
import tensorflow as tf
from keras import saving
from utility.models import ModelHelper

class GeneratorModel(models.Model):
    def __init__(self, i_shape:int, learning_rate, higher_images_resolution=True):
        super().__init__()
        self._bin_loss = losses.BinaryCrossentropy()
        self._loc_optimizer = optimizers.Adam(learning_rate)
        self._loss_tracker = metrics.Mean(name="loss")
        self._noise_vector_shape = i_shape
        self._higher_images_resolution = higher_images_resolution
        
        self._dense_layer_1 = layers.Dense(64, activation='relu', input_shape=i_shape,
            kernel_initializer='he_normal') 
        self._dense_layer_2 = layers.Dense(512, activation='relu', 
            kernel_initializer='he_normal')
        self._reshape_layer_1 = layers.Reshape((8, 8, -1))

        self._conv_block_1 = ModelHelper.Conv2D_T_BlockBuilder.construct(128, (3,3))
        self._conv_block_2 = ModelHelper.Conv2D_T_BlockBuilder.construct(64 , (3,3))
        self._conv_block_3 = ModelHelper.Conv2D_T_BlockBuilder.construct(32 , (3,3))
        
        if self._higher_images_resolution: # (128,128)
            self._conv_block_4 = ModelHelper.Conv2D_T_BlockBuilder.construct(16 , (3,3))
        
        self._conv_layer_1 = layers.Conv2D(3, (1,1), activation='sigmoid',
            kernel_initializer='glorot_normal')

    def compute_loss(self, gen_images_output):
        fake_images_true_value = tf.ones_like(gen_images_output)
        return self._bin_loss(fake_images_true_value, gen_images_output)

    @property
    def noise_vector_shape(self):
        return self._noise_vector_shape
    
    @property
    def learning_rate(self):
        return self._loc_optimizer.learning_rate.numpy()

    @property
    def image_higher_resolution(self):
        return self._higher_images_resolution

    def call(self, inputs, training=None):
        x = self._dense_layer_1(inputs, training=training)
        x = self._dense_layer_2(x, training=training)
        x = self._reshape_layer_1(x, training=training)
        x = ModelHelper.LayerIterator(self._conv_block_1)(x, training)
        x = ModelHelper.LayerIterator(self._conv_block_2)(x, training)      
        x = ModelHelper.LayerIterator(self._conv_block_3)(x, training)

        if self._higher_images_resolution:
            x = ModelHelper.LayerIterator(self._conv_block_4)(x, training)

        x = self._conv_layer_1(x, training=training)

        return x

    def apply_grads(self, gen_grads):
        self._loc_optimizer.apply_gradients(zip(gen_grads, self.trainable_variables))

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