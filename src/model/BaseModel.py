from keras import models
from keras import saving
from keras import metrics
import tensorflow as tf
from .  import GeneratorModel as gen_model, DiscriminatorModel as disc_model
from keras import callbacks
from callbacks import GeneratorCallback as gen_callback,\
    CheckpointCallback as ckpt_callback, ModelCallback as mod_callback
from os import path
from utility.dataset import ImageProvider as img_provider
from keras import optimizers
from keras import export
from utility.enums.defs import ImageResolutionOps

@saving.register_keras_serializable(package="Model", name="Base")
class BaseModel(models.Model):
    def __init__(self, 
                 g_learning_rate : float | optimizers.schedules.LearningRateSchedule, 
                 d_learning_rate : float | optimizers.schedules.LearningRateSchedule,
                 noise_vector, batch_size, d_input):
        super().__init__()
        ImageResolutionOps.loc_range(d_input[:2])
        self._generator_model = gen_model.GeneratorModel(noise_vector, g_learning_rate)
        self._discriminator_model = disc_model.DiscriminatorModel(d_input, d_learning_rate)
        self._batch_size = batch_size
        self._callback_list = callbacks.CallbackList(add_history=True, add_progbar=True)

    def get_config(self):
        architecture_configuration = {
            "g_learning_rate" : self._generator_model.learning_rate,
            "d_learning_rate" : self._discriminator_model.learning_rate,
            "noise_vector" : self._generator_model.noise_vector_shape,
            "batch_size" : self._batch_size,
            "d_input" : self._discriminator_model.shape
        }
        return architecture_configuration
    
    def generate(self, noise_vectors):
        return self._generator_model(noise_vectors, training=False)

    @classmethod
    def from_config(cls, config):
        g_learning_rate = config.pop("g_learning_rate")
        d_learning_rate = config.pop("d_learning_rate")
        noise_vector    = config.pop("noise_vector")
        batch_size      = config.pop("batch_size")
        d_input         = config.pop("d_input")

        return cls(g_learning_rate, d_learning_rate, noise_vector, 
                   batch_size, d_input)
    
    def get_build_config(self):
        g_build_config = self._generator_model.get_build_config()
        d_build_config = self._discriminator_model.get_build_config()
        return {
            "g_build_configuration" : g_build_config,
            "d_build_configuration" : d_build_config
        }
    
    def get_compile_config(self):
        g_compile_config = self._generator_model.get_compile_config()
        d_compile_config = self._discriminator_model.get_compile_config()
        return {
            "g_compile_configuration" : g_compile_config,
            "d_compile_configuration" : d_compile_config
        }

    def build_from_config(self, config):
        g_build_config = config.pop("g_build_configuration")
        d_build_config = config.pop("d_build_configuration")
        self._generator_model.build_from_config(g_build_config)
        self._discriminator_model.build_from_config(d_build_config)

    def compile_from_config(self, config):
        g_compile_config = config.pop("g_compile_configuration")
        d_compile_config = config.pop("d_compile_configuration")
        self._generator_model.compile_from_config(g_compile_config)
        self._discriminator_model.compile_from_config(d_compile_config)

    @tf.function
    def train_step(self, batch):
        noise_vector_shape = self._generator_model.noise_vector_shape
        batch_noise_vector = tf.random.normal([self._batch_size, noise_vector_shape])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self._generator_model(batch_noise_vector, training=True)
            fake_out = self._discriminator_model(fake_images, training=True)
            real_out = self._discriminator_model(batch, training=True)
            generator_loss = self._generator_model.compute_loss(fake_out)
            discriminator_loss = self._discriminator_model.compute_loss(fake_out,real_out)

        gen_grads = g_tape.gradient(generator_loss, self._generator_model.trainable_variables)
        self._generator_model.apply_grads(gen_grads)

        discr_grads = d_tape.gradient(discriminator_loss, self._discriminator_model.trainable_variables)
        self._discriminator_model.apply_grads(discr_grads)

        return {"gen_loss" : generator_loss, "disc_loss" : discriminator_loss}

    # this function is just used inside the model callback -> build model so that we can gather
    # model information
    def call(self, inputs):
        noise_vector_shape = self._generator_model.noise_vector_shape
        batch_noise_vector = tf.random.normal([self._batch_size, noise_vector_shape])
        generated_images = self._generator_model(batch_noise_vector, training=False)
        self._discriminator_model(generated_images, training=False)

    def fit(self, dataset, generate_images_per_epoch, epochs=100, with_early_stop=False, 
            generate_while_training=True):
        # history callback is passed in the constructor of the callback list.
        early_stop_callback1 = callbacks.EarlyStopping("gen_loss", patience=5, mode="min", start_from_epoch=10) # stop crit for generator
        early_stop_callback2 = callbacks.EarlyStopping("disc_loss", patience=5, mode="min", start_from_epoch=10) # stop crit for discriminator
        """ we don't want to save each epoch -> write custom saving callback
        checkpoint_callback1 = callbacks.ModelCheckpoint( 
            path.join("ckpt", "weights", "ckpt.weights.{epoch:02d}.h5"),
            save_freq='epoch') # only saving weights """
        
        # since base model contains the *from_config() it is necassary to revert base model status
        self.compiled = False
        self.built    = False

        image_provider = img_provider.ImageProvider.build_from_dataset(
            self._batch_size, dataset)
        img_h, img_w, _ = image_provider.provide_image_dim()
        checkpoint_callback1 = ckpt_callback.CheckpointCallback()
        checkpoint_callback2 = callbacks.ModelCheckpoint(
            path.join("ckpt", "model", f"model_tgt_{img_h}-{img_w}.keras"),
            save_freq='epoch', save_weights_only=False) # saving whole model
        generator_callback = gen_callback.GeneratorCallback(
            image_provider, generate_images_per_epoch)
        model_callback = mod_callback.ModelCallback()

        if(with_early_stop):
            self._callback_list.append(early_stop_callback1)
            self._callback_list.append(early_stop_callback2)

        if(generate_while_training):
            self._callback_list.append(generator_callback)
        self._callback_list.append(checkpoint_callback1)
        self._callback_list.append(checkpoint_callback2)
        self._callback_list.append(model_callback)
        
        self._callback_list.set_model(self)
        self._callback_list.set_params({"epochs" : epochs, "verbose" : 1,
            "steps" : len(dataset)})
        
        self._callback_list.on_train_begin()

        for e in range(epochs):
            #if(self.stop_training): # early stopping criteria has been met
            #    return self.history
            self._callback_list.on_epoch_begin(e)
            batch_nr = 1

            for batch in dataset:
                self._callback_list.on_train_batch_begin(batch_nr)
                loss_dict = self.train_step(batch)
                self._generator_model.metrics[0].update_state(loss_dict["gen_loss"])
                self._discriminator_model.metrics[0].update_state(loss_dict["disc_loss"])
                self._callback_list.on_train_batch_end(batch_nr)
                batch_nr += 1

            self._callback_list.on_epoch_end(e, {
                "gen_loss" : self._generator_model.metrics[0].result(),
                "disc_loss" : self._discriminator_model.metrics[0].result()})
        
            self._generator_model.metrics[0].reset_state()
            self._discriminator_model.metrics[0].reset_state()

        self._callback_list.on_train_end()

        return self.history

    def evaluate(self, quantity:int=1000):
        noise_vector_shape = self._generator_model.noise_vector_shape
        _range = quantity // self._batch_size 
        bic_generator = metrics.BinaryCrossentropy()
        bic_discriminator = metrics.BinaryCrossentropy()
        bia = metrics.BinaryAccuracy()
        for _ in range(_range):
            noise_vectors = tf.random.normal([self._batch_size, noise_vector_shape])
            generated_images  = self._generator_model(noise_vectors, training=False)
            discriminator_out = self._discriminator_model(generated_images, training=False)
            generator_expected_values     = tf.ones_like(discriminator_out)
            discriminator_expected_values = tf.zeros_like(discriminator_out)
            bic_generator.update_state(generator_expected_values, discriminator_out)
            bic_discriminator.update_state(discriminator_expected_values, discriminator_out)
            bia.update_state(generator_expected_values, discriminator_out)
        
        gen_acc = bia.result()
        return {"gen_loss" : bic_generator.result(), "disc_loss" : bic_discriminator.result(), 
                "gen_acc" : gen_acc, "disc_acc" : 1 - gen_acc}
    
    def predict(self, quantity:int=5):
        noise_vector_shape = self._generator_model.noise_vector_shape
        noise_vectors = tf.random.normal([quantity, noise_vector_shape])
        return self._generator_model(noise_vectors, training=False)
    
    def checkpoint(self, _ckpt):
        path_str = path.join("ckpt", "weights", f"ckpt_{_ckpt}.weights.h5")
        self.save_weights(path_str, overwrite=True)

    def apply_summary(self, p_fn):
        self.summary(print_fn=p_fn)
        self._generator_model.apply_summary(p_fn)
        self._discriminator_model.apply_summary(p_fn)

    def export(self):
        gen_dict  = self.generator
        img_shape = self.discriminator["d_input"]
        ex_arch = export.ExportArchive()
        ex_arch.track(self._generator_model)
        ex_arch.add_endpoint("inference", self._generator_model.call,
            input_signature=[tf.TensorSpec(
                shape=(None, gen_dict["noise_vector"]), dtype=tf.float32)])
        ex_arch.write_out(path.join("ckpt", "model", "inference", 
            f"inf_v_{img_shape[0]}-{img_shape[1]}"))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def discriminator(self):
        return {
            "d_input" : self._discriminator_model.shape,
            "learning_rate" : self._discriminator_model.learning_rate 
        }

    @property
    def generator(self):
        return {
            "noise_vector" : self._generator_model.noise_vector_shape,
            "learning_rate" : self._generator_model.learning_rate,
        }