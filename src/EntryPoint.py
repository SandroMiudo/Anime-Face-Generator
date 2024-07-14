from model import BaseModel as base_model, DiscriminatorModel as discriminator_model, \
    GeneratorModel as gen_model
from utility.dataset import ImageProvider as img_provider
from keras import optimizers
from keras import metrics
from keras import losses

provider = img_provider.ImageProvider()
dataset, batch_size = provider.provide_images()
input_shape = provider.provide_image_dim()
d_model  = discriminator_model.DiscriminatorModel(input_shape)
g_model  = gen_model.GeneratorModel()
b_model  =  base_model.BaseModel(g_model, d_model, batch_size)

d_model.compile(optimizers.Adam(0.0001), losses.BinaryCrossentropy(from_logits=True),
    metrics=[metrics.Mean()])
g_model.compile(optimizers.Adam(0.0001), losses.BinaryCrossentropy(from_logits=True),
    metrics=[metrics.Mean()])

epochs = 5

history = b_model.fit(dataset, epochs)

images_generated = b_model.predict()