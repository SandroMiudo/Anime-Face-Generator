from model import BaseModel as base_model, DiscriminatorModel as discriminator_model, GeneratorModel as gen_model
from utility.dataset import ImageProvider as img_provider

provider = img_provider.ImageProvider()
dataset, batch_size = provider.provide_images()
input_shape = provider.provide_image_dim()
d_model  = discriminator_model.DiscriminatorModel(input_shape)
g_model  = gen_model.GeneratorModel()
b_model  =  base_model.BaseModel(g_model, d_model)