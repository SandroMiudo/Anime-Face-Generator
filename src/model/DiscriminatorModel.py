from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
import tensorflow as tf
from keras import saving
from utility.models.ModelHelper import Conv2DBlockBuilder, LayerIterator
from utility.enums.defs import ImageResolution
from utility.enums.defs import ImageResolutionOps

class DiscriminatorModel(models.Model):

    def __init__(self, i_shape:tuple[int,int,int], learning_rate):
        super().__init__()
        self._i_shape = i_shape
        self._bin_loss = losses.BinaryCrossentropy()
        self._loc_optimizer = optimizers.Adam(learning_rate)
        self._loss_tracker = metrics.Mean(name="loss")

        _kernel_c = (5,5)

        self._conv_block_1 = Conv2DBlockBuilder.construct(32, _kernel_c)
        self._conv_block_2 = Conv2DBlockBuilder.construct(64, _kernel_c)

        if ImageResolutionOps.tgt_in(ImageResolution.x_96_96):
            self._conv_block_3 = Conv2DBlockBuilder.construct(96, _kernel_c)

        if ImageResolutionOps.tgt_in(ImageResolution.x_112_112):
            self._conv_block_3 = Conv2DBlockBuilder.construct(112, _kernel_c)

        if ImageResolutionOps.tgt_in(ImageResolution.x_128_128):
            self._conv_block_3 = Conv2DBlockBuilder.construct(128, _kernel_c)

        if ImageResolutionOps.tgt_in(ImageResolution.x_256_256):
            self._conv_block_4 = Conv2DBlockBuilder.construct(256, _kernel_c)

        if ImageResolutionOps.tgt_in(ImageResolution.x_512_512):
            self._conv_block_5 = Conv2DBlockBuilder.construct(512, _kernel_c)

        self._flatten_layer = layers.Flatten()
        self._dense_layer_1 = layers.Dense(128, activation='leaky_relu', 
            kernel_initializer='he_normal')
        self._dense_layer_2 = layers.Dense(1, activation='sigmoid',
            kernel_initializer='glorot_normal')

    def compute_loss(self, gen_images_output, real_images):
        real_images_true_value = tf.ones_like(real_images) * 0.9
        fake_images_true_value = tf.zeros_like(gen_images_output) + 0.1

        return self._bin_loss(real_images_true_value, real_images) + self._bin_loss(fake_images_true_value, gen_images_output)

    def call(self, inputs, training=None):
        _noise = tf.random.normal(list(inputs.shape), 0, 0.1) # potentialy increase this to 0.15 - 0.2
        inputs = tf.clip_by_value(inputs + _noise, 0.0, 1.0)

        x = LayerIterator(self._conv_block_1)(inputs, training)
        x = LayerIterator(self._conv_block_2)(x, training)

        if ImageResolutionOps.tgt_in(ImageResolution.x_96_96) or \
           ImageResolutionOps.tgt_in(ImageResolution.x_112_112) or \
           ImageResolutionOps.tgt_in(ImageResolution.x_128_128):
            x = LayerIterator(self._conv_block_3)(x, training)

        if ImageResolutionOps.tgt_in(ImageResolution.x_256_256):
            x = LayerIterator(self._conv_block_4)(x, training)
        
        if ImageResolutionOps.tgt_in(ImageResolution.x_512_512):
            x = LayerIterator(self._conv_block_5)(x, training)

        x = self._flatten_layer(x)
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