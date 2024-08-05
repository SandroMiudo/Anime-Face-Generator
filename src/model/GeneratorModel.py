from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
import tensorflow as tf
from keras import saving
from utility.models.ModelHelper import Conv2D_T_BlockBuilder, LayerIterator
from utility.enums.defs import ImageResolution
from utility.enums.defs import ImageResolutionOps

class GeneratorModel(models.Model):
    def __init__(self, i_shape:int, learning_rate):
        super().__init__()
        self._bin_loss = losses.BinaryCrossentropy()
        self._loc_optimizer = optimizers.Adam(learning_rate)
        self._loss_tracker = metrics.Mean(name="loss")
        self._noise_vector_shape = i_shape
        
        _reshape_tpl = ()
        _kernel_c_t = (2,2)

        if ImageResolutionOps.tgt_in(ImageResolution.x_64_64):
            _reshape_tpl = (8, 8, 256)

        elif ImageResolutionOps.tgt_in(ImageResolution.x_56_56):
            _reshape_tpl = (7, 7, 256)

        elif ImageResolutionOps.tgt_in(ImageResolution.x_48_48): 
            _reshape_tpl = (6, 6, 256)

        self._reshape_layer_1 = layers.Reshape(_reshape_tpl)

        self._dense_layer_1 = layers.Dense(_reshape_tpl[0]*_reshape_tpl[1]*_reshape_tpl[2],
            activation='leaky_relu', kernel_initializer='he_normal', input_shape=i_shape)

        if ImageResolutionOps.tgt_in(ImageResolution.x_512_512):
            self._conv_x_512 = Conv2D_T_BlockBuilder.construct(512, _kernel_c_t)

        if ImageResolutionOps.tgt_in(ImageResolution.x_256_256):
            self._conv_x_256 = Conv2D_T_BlockBuilder.construct(256, _kernel_c_t)

        if ImageResolutionOps.tgt_in(ImageResolution.x_128_128):
            self._conv_x_128 = Conv2D_T_BlockBuilder.construct(128, _kernel_c_t)
        
        if ImageResolutionOps.tgt_in(ImageResolution.x_112_112):
            self._conv_x_112 = Conv2D_T_BlockBuilder.construct(112, _kernel_c_t)

        if ImageResolutionOps.tgt_in(ImageResolution.x_96_96):
            self._conv_x_96  = Conv2D_T_BlockBuilder.construct(96, _kernel_c_t)

        self._conv_block_1 = Conv2D_T_BlockBuilder.construct(64, _kernel_c_t)
        self._conv_block_2 = Conv2D_T_BlockBuilder.construct(32, _kernel_c_t)
        self._conv_layer_3 = layers.Conv2DTranspose(16, _kernel_c_t, (2,2), padding='same')
        self._conv_layer_4 = layers.Conv2D(3, (1,1), activation='sigmoid',
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

    def call(self, inputs, training=None):
        x = self._dense_layer_1(inputs, training=training)
        x = self._reshape_layer_1(x, training=training)

        if ImageResolutionOps.tgt_in(ImageResolution.x_512_512):
            x = LayerIterator(self._conv_x_512)(x, training)

        if ImageResolutionOps.tgt_in(ImageResolution.x_256_256):
            x = LayerIterator(self._conv_x_256)(x, training)

        if ImageResolutionOps.tgt_in(ImageResolution.x_128_128):
            x = LayerIterator(self._conv_x_128)(x, training)

        if ImageResolutionOps.tgt_in(ImageResolution.x_112_112):
            x = LayerIterator(self._conv_x_112)(x, training)

        if ImageResolutionOps.tgt_in(ImageResolution.x_96_96):
            x = LayerIterator(self._conv_x_96)(x, training)

        x = LayerIterator(self._conv_block_1)(x, training=training)
        x = LayerIterator(self._conv_block_2)(x, training=training)
        x = self._conv_layer_3(x, training=training)
        x = self._conv_layer_4(x, training=training)

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