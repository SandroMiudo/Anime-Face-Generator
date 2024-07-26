from keras import callbacks
from PIL import Image
import numpy as np
from utility.plot import DiagramPlotter as dg_plt
from utility.plot import ImagePlotter as img_plt
from os import path

class GeneratorCallback(callbacks.Callback):
    def __init__(self, image_provider, generate_images_per_epoch=1, generate_seed_images=2):
        super().__init__()
        self._image_plotter = img_plt.ImagePlotter(image_provider)
        self._generate_images_per_epoch = generate_images_per_epoch
        self._generate_seed_images      = generate_seed_images

    def on_epoch_end(self, epoch, logs=None):
        self.generate_image(epoch, True, self._generate_images_per_epoch)
        self.generate_image(epoch, False, self._generate_seed_images)

    def generate_image(self, epoch, rand: bool, generate_c: int):
        noise_vectors = None
        out_loc       = None
        if(rand):
            noise_vectors = np.random.normal(0, 1, [self._generate_images_per_epoch, 
            self._noise_vector])
            out_loc = path.join("generator_out", "random", f"generated_in_epoch_{epoch+1}_img_")
        else:
            noise_vectors = self._seed
            out_loc = path.join("generator_out", "seed", f"generated_in_epoch_{epoch+1}_seed_img_")
        generated_images = self.model.generate(noise_vectors)
        for i in range(generate_c):
            tmp_loc = out_loc + f"{i+1}.jpg"
            print(tmp_loc)
            self.image_routine(generated_images[i], tmp_loc)
    
    def image_routine(self, generated_image, out_loc):
        image_result = Image.fromarray(np.asarray(generated_image * 255, dtype=np.uint8))
        image_result.save(out_loc)

    def on_train_begin(self, logs=None):
        # self._image_plotter.plot()
        # self._image_plotter.plot("denorm")
        generator_properties = self.model.generator
        self._noise_vector = generator_properties["noise_vector"]
        self._seed = np.random.normal(0, 1, [self._generate_seed_images, self._noise_vector])

    def on_train_end(self, logs=None):
        diagramm_plotter = dg_plt.DiagramPlotter(self.model.history)
        diagramm_plotter.plot()