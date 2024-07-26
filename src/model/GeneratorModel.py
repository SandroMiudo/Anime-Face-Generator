from keras import models
from keras import layers
from keras import initializers
from keras import losses
from keras import metrics
from keras import regularizers
from keras import optimizers
import tensorflow as tf
from keras import saving

class GeneratorModel(models.Model):
    # try setting the noise vector higher and the connecting to the first dense layer to fewer units
    def __init__(self, i_shape:int, learning_rate, higher_images_resolution=True):
        super().__init__()
        self._bin_loss = losses.BinaryCrossentropy()
        self._loc_optimizer = optimizers.Adam(learning_rate)
        self._loss_tracker = metrics.Mean(name="loss")
        self._noise_vector_shape = i_shape
        self._higher_images_resolution = higher_images_resolution
        
        self._dense_layer_1 = layers.Dense(128, activation='tanh', input_shape=i_shape,
            kernel_regularizer=regularizers.l2(0.01))

        self._dense_layer_2 = layers.Dense(256, activation='tanh',
            kernel_regularizer=regularizers.l2(0.01))

        self._reshape_layer_1 = layers.Reshape((4, 4, -1))

        self._conv_layer_1 = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',
            activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self._batch_normalize_layer_1 = layers.BatchNormalization()

        self._conv_layer_2 = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same',
            activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self._batch_normalize_layer_2 = layers.BatchNormalization()

        self._conv_layer_3 = layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same',
            activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self._batch_normalize_layer_3 = layers.BatchNormalization()

        self._conv_layer_4 = layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same',
            activation='relu', kernel_regularizer=regularizers.l2(0.01))
        self._batch_normalize_layer_4 = layers.BatchNormalization()

        if self._higher_images_resolution: # (128,128)
            self._conv_layer_5 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same',
                activation='sigmoid')
        else: # (64,64)
            self._conv_layer_5 = layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same')

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
        x = self._dense_layer_2(inputs, training=training)
        x = self._reshape_layer_1(x, training=training)
        x = self._conv_layer_1(x, training=training)
        x = self._batch_normalize_layer_1(x, training=training)
        x = self._conv_layer_2(x, training=training)
        x = self._batch_normalize_layer_2(x, training=training)
        x = self._conv_layer_3(x, training=training)
        x = self._batch_normalize_layer_3(x, training=training)
        x = self._conv_layer_4(x, training=training)
        x = self._batch_normalize_layer_4(x, training=training)

        if(self._higher_images_resolution):
            x = self._conv_layer_5(x, training=training)
        else:
            x = self._conv_layer_5(x, training=training)

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