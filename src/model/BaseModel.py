from keras import models
from keras import saving

@saving.register_keras_serializable()
class BaseModel(models.Model):
    def __init__(self, generator_model, discriminator_model):
        super().__init__()
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def get_config(self):
        return super().get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        return super().from_config(config, custom_objects)