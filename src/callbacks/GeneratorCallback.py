from keras import callbacks
from PIL import Image
import numpy as np

class GeneratorCallback(callbacks.Callback):
    def __init__(self, generate_images_per_epoch=1):
        super().__init__()
        self.generate_images_per_epoch = generate_images_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generate(self.generate_images_per_epoch)
        image_result = Image.fromarray(np.asarray(generated_images[0, ...] * 127.5 + 127.5, dtype=np.uint8))
        image_result.save(f"generator_out/gen_epoch_{epoch}.jpg")