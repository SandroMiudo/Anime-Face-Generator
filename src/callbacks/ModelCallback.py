from keras.callbacks import Callback
from keras import utils
from os import path

class ModelCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        path_to_file = path.join("media", "model_summary.txt")
        with open(path_to_file, 'w'):
            pass

        def write_fn(line):
            with open(path_to_file, 'a') as f:
                f.write(line)

        model = self.model
        model(None) # build model
        self.model.apply_summary(write_fn)