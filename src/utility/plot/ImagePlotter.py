import matplotlib.pyplot as plt
from . import Plotter as pl
import numpy as np
import os
from ..dataset import ImageProvider as img_provider
import time

class ImagePlotter(pl.Plotter):
    def __init__(self, image_provider:img_provider.ImageProvider, 
                 axes: tuple[int, int]=(4,4)):
        super().__init__(axes, xticks=[], yticks=[])
        self.image_provider = image_provider

    def plot(self, *args):
        axes_shape = self.axes.shape
        denormalize = False
        if('denorm' in args):
            denormalize = True
        for i in range(axes_shape[0]):
            for j in range(axes_shape[1]):
                image, _ = self.image_provider.sample_images(time.time_ns(), 1)
                self.axes[i, j].axis('off')
                if(not denormalize):
                    self.axes[i, j].imshow(np.asarray(image[0], dtype=float))
                else:
                    if(img_provider.ImageProvider.normalize_function_used == 
                       img_provider.ImageProvider.normalize_image_minus_one_to_one):
                        self.axes[i, j].imshow(np.asarray(image[0] * 127.5 + 127.5, dtype=int))
                    else:
                        self.axes[i, j].imshow(np.asarray(image[0] * 255, dtype=int))
        if(not denormalize):
            self.save(os.path.join("media", "images_normalized.png"))
        else:
            self.save(os.path.join("media", "images.png"))