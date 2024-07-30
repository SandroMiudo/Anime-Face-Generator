import matplotlib.pyplot as plt
from . import Plotter as pl
import numpy as np
import os
from ..dataset import ImageProvider as img_provider
import time
from typing import Any

class ImagePlotter(pl.Plotter):
    def __init__(self, image_provider:img_provider.ImageProvider | None, 
                 axes: tuple[int, int]=(4, 4)):
        super().__init__(axes, xticks=[], yticks=[])
        self._image_provider = image_provider

    def plot(self, *args):
        axes_shape = self.axes.shape
        denormalize = False
        if('denorm' in args):
            denormalize = True
        for i in range(axes_shape[0]):
            for j in range(axes_shape[1]):
                image, _ = self._image_provider.sample_images(time.time_ns(), 1)
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

    def plot_from_dataset(self, dataset, k, index, show=True):
        if(k >= self.axes.shape[0] * self.axes.shape[1]):
            return
        if(index == -1):
            index = time.time_ns() % len(dataset)
        if(index != 0):
            dataset = dataset.skip(index-1)
        row=k//self.axes.shape[1]
        col=k %self.axes.shape[1]
        d_iterator = iter(dataset)
        image = d_iterator.get_next()
        self.axes[row, col].axis('off')
        self.axes[row, col].imshow(np.clip(np.asarray(image), [0], [1]))

        if(show):
            plt.show()

    def plot_from_datasets(self, datasets : list[Any], plot_per_dataset=2):
        k=0
        for dataset in datasets:
            for _ in range(plot_per_dataset):
                self.plot_from_dataset(dataset, k, -1, show=False)
                k+=1
        plt.show()

    def plot_from_image(self, image):
        self.axes[0, 0].axis('off')
        self.axes[0, 0].imshow(image)
        plt.show()