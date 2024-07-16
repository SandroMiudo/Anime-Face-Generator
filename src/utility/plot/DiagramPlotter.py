import matplotlib.pyplot as plt
from . import Plotter as pl
from keras.callbacks import History
import numpy as np
import os

class DiagramPlotter(pl.Plotter):
    def __init__(self, history:History, axes: tuple[int, int]=(1,1)):
        super().__init__(axes)
        self.history = history

    def plot(self):
        epochs = self.history.epoch
        d_loss = self.history.history["gen_loss"]
        g_loss = self.history.history["disc_loss"]

        self.axes.axis('on')
        self.axes.set_xlim(0, len(epochs))
        self.axes.set_ylim(0, 10)
        self.axes.set_xlabel("epochs")
        self.axes.set_ylabel("loss")
        self.axes.set_title("loss over time")
        self.axes.set_xticks(np.arange(0, len(epochs), 10))

        self.axes.plot(epochs, g_loss, '-b', epochs, d_loss, '-r')
        self.axes.legend(['Generator Loss', 'Discriminator Loss'], loc='upper right')

        self.save(os.path.join("plots", "loss_plot.png"))