from keras import models
from keras import saving
import tensorflow as tf
from .  import GeneratorModel as gen_model, DiscriminatorModel as disc_model
from keras import callbacks
from callbacks import GeneratorCallback as gen_callback, CheckpointCallback as ckpt_callback
from os import path
from utility.dataset import ImageProvider as img_provider

@saving.register_keras_serializable()
class BaseModel(models.Model):
    def __init__(self, generator_model:gen_model.GeneratorModel, 
                 discriminator_model:disc_model.DiscriminatorModel,
                 batch_size: int):
        super().__init__()
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.batch_size = batch_size
        self.callback_list = callbacks.CallbackList(add_history=True, add_progbar=True)

    def get_config(self):
        base_configuration = super().get_config()
        gen_model_serialization  = self.generator_model.get_config()
        disc_model_serialization = self.discriminator_model.get_config()

        architecture_configuration = {
            "generator_model" : gen_model_serialization,
            "discriminator_model" : disc_model_serialization,
            "batch_size" : self.batch_size
        }

        return {**base_configuration, **architecture_configuration}
    
    def generate(self, n):
        noise_vectors = tf.random.normal([n, 
            self.generator_model.get_noise_vector_shape()])
        return self.generator_model(noise_vectors, training=False)

    @classmethod
    def from_config(cls, config):
        gen_model_serialization  = config.pop("generator_model")
        disc_model_serialization = config.pop("discriminator_model")

        generator_model = gen_model.GeneratorModel.from_config(gen_model_serialization)
        discriminator_model = disc_model.DiscriminatorModel.from_config(disc_model_serialization)
        return cls(generator_model, discriminator_model, config.pop("batch_size"))
    
    @tf.function
    def train_step(self, batch):
        noise_vector_shape = self.generator_model.get_noise_vector_shape()
        batch_noise_vector = tf.random.normal([self.batch_size, noise_vector_shape])

        with tf.GradientTape(persistent=True) as tape:
            fake_images = self.generator_model(batch_noise_vector, training=True)
            fake_out = self.discriminator_model(fake_images, training=True)
            real_out = self.discriminator_model(batch, training=True)
            generator_loss = self.generator_model.compute_loss(fake_out)
            discriminator_loss = self.discriminator_model.compute_loss(fake_out,real_out)

        gen_grads = tape.gradient(generator_loss, self.generator_model.trainable_variables)
        self.generator_model.apply_grads(gen_grads)

        discr_grads = tape.gradient(discriminator_loss, self.discriminator_model.trainable_variables)
        self.discriminator_model.apply_grads(discr_grads)

        del tape

        return {"gen_loss" : generator_loss, "disc_loss" : discriminator_loss}

    def fit(self, dataset, generate_images_per_epoch, epochs=100, with_early_stop=False, 
            generate_while_training=True):
        # history callback is passed in the constructor of the callback list.
        early_stop_callback1 = callbacks.EarlyStopping("gen_loss", patience=5, mode="min", start_from_epoch=10) # stop crit for generator
        early_stop_callback2 = callbacks.EarlyStopping("disc_loss", patience=5, mode="min", start_from_epoch=10) # stop crit for discriminator
        """ we don't want to save each epoch -> write custom saving callback
        checkpoint_callback1 = callbacks.ModelCheckpoint( 
            path.join("ckpt", "weights", "ckpt.weights.{epoch:02d}.h5"),
            save_freq='epoch') # only saving weights """
        
        image_provider = img_provider.ImageProvider.build_from_dataset(self.batch_size,
                                                                       dataset)
        checkpoint_callback1 = ckpt_callback.CheckpointCallback()
        checkpoint_callback2 = callbacks.ModelCheckpoint(
            path.join("ckpt", "model", "model.keras"),
            save_freq='epoch', save_weights_only=False) # saving whole model
        generator_callback = gen_callback.GeneratorCallback(image_provider, generate_images_per_epoch)

        if(with_early_stop):
            self.callback_list.append(early_stop_callback1)
            self.callback_list.append(early_stop_callback2)

        if(generate_while_training):
            self.callback_list.append(generator_callback)
        self.callback_list.append(checkpoint_callback1)
        self.callback_list.append(checkpoint_callback2)
        
        self.callback_list.set_model(self)
        self.callback_list.set_params({"epochs" : epochs, "verbose" : 1,
            "steps" : len(dataset)})

        self.callback_list.on_train_begin()

        for e in range(epochs):
            #if(self.stop_training): # early stopping criteria has been met
            #    return self.history
            self.callback_list.on_epoch_begin(e)
            batch_nr = 1

            for batch in dataset:
                self.callback_list.on_train_batch_begin(batch_nr)
                loss_dict = self.train_step(batch)
                self.generator_model.metrics[0].update_state(loss_dict["gen_loss"])
                self.discriminator_model.metrics[0].update_state(loss_dict["disc_loss"])
                self.callback_list.on_train_batch_end(batch_nr)
                batch_nr += 1

            self.callback_list.on_epoch_end(e, {
                "gen_loss" : self.generator_model.metrics[0].result(),
                "disc_loss" : self.discriminator_model.metrics[0].result()})
        
            self.generator_model.metrics[0].reset_state()
            self.discriminator_model.metrics[0].reset_state()

        self.callback_list.on_train_end()

        return self.history

    def evaluate(self, quantity:int=1000):
        noise_vector_shape = self.generator_model.get_noise_vector_shape()
        _range = self.batch_size // quantity
        bic_generator = tf.metrics.BinaryCrossentropy(from_logits=True)
        bic_discriminator = tf.metrics.BinaryCrossentropy(from_logits=True)
        bia = tf.metrics.BinaryAccuracy()
        for _ in range(_range):
            noise_vectors = tf.random.normal([self.batch_size, noise_vector_shape])
            generated_images  = self.generator_model(noise_vectors, training=False)
            discriminator_out = self.discriminator_model(generated_images, training=False)
            generator_expected_values     = tf.ones_like(discriminator_out)
            discriminator_expected_values = tf.zeros_like(discriminator_out)
            bic_generator.update_state(generator_expected_values, discriminator_out)
            bic_discriminator.update_state(discriminator_expected_values, discriminator_out)
            bia.update_state(generator_expected_values, discriminator_out)
        
        gen_acc = bia.result()
        return {"gen_loss" : bic_generator.result(), "disc_loss" : bic_discriminator.result(), 
                "gen_acc" : gen_acc, "disc_acc" : 1 - gen_acc}
    
    """
    used to generate images -> inference
    """
    def predict(self, quantity:int=5):
        noise_vector_shape = self.generator_model.get_noise_vector_shape()
        noise_vectors = tf.random.normal([quantity, noise_vector_shape])
        return self.generator_model(noise_vectors, training=False)
    
    def checkpoint(self, _ckpt):
        path_str = path.join("ckpt", "weights", f"ckpt_{_ckpt}.weights.h5")
        self.save_weights(path_str, overwrite=True)