from keras import callbacks
from PIL import Image
import numpy as np
from utility.plot import DiagramPlotter as dg_plt
from utility.plot import ImagePlotter as img_plt

class GeneratorCallback(callbacks.Callback):
    def __init__(self, image_provider, generate_images_per_epoch=1):
        super().__init__()
        self.image_plotter = img_plt.ImagePlotter(image_provider)
        self.generate_images_per_epoch = generate_images_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.generate_images_per_epoch):
            self.generate_image(epoch, i)

    def generate_image(self, epoch, i):
        generated_images = self.model.generate(self.generate_images_per_epoch)
        image_result = Image.fromarray(np.asarray(generated_images[0, ...] * 127.5 + 127.5, dtype=np.uint8))
        image_result.save(f"generator_out/gen_epoch_{epoch}_{i}.jpg")
    
    def on_train_begin(self, logs=None):
        self.image_plotter.plot()
        self.image_plotter.plot("denorm")

    def on_train_end(self, logs=None):
        diagramm_plotter = dg_plt.DiagramPlotter(self.model.history)
        diagramm_plotter.plot()