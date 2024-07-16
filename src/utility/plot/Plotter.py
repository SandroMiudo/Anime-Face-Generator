import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, axes:tuple[int,int], **kwargs):
        fig, axes = plt.subplots(axes[0], axes[1], subplot_kw=kwargs)
        self.figure = fig
        self.axes   = axes

    def plot(self, *args):
        pass

    def save(self, loc):
        self.figure.savefig(loc, format='png')