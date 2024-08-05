import matplotlib.pyplot as plt
from . import Plotter as pl
from keras.callbacks import History
import numpy as np
import os

class DiagramPlotter(pl.Plotter):
    def __init__(self, history:History, axes: tuple[int, int]=(1,1)):
        super().__init__(axes)
        self._history = history

    def plot(self):
        epochs = self._history.epoch
        g_loss = self._history.history["gen_loss"]
        d_loss = self._history.history["disc_loss"]

        self.axes[0,0].axis('on')
        self.axes[0,0].set_xlim(0, len(epochs))
        self.axes[0,0].set_ylim(0, 10)
        self.axes[0,0].set_xlabel("epochs")
        self.axes[0,0].set_ylabel("loss")
        self.axes[0,0].set_title("loss over time")
        self.axes[0,0].set_xticks(np.arange(0, len(epochs), 10))

        self.axes[0,0].plot(epochs, g_loss, '-b', epochs, d_loss, '-r')
        self.axes[0,0].legend(['Generator Loss', 'Discriminator Loss'], loc='upper right')

        self.save(os.path.join("plots", "loss_plot.png"))