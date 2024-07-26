from keras.callbacks import Callback
from keras import optimizers
from os import path

class ModelCallback(Callback):
    def __init__(self):
        super().__init__()

    def set_params(self, params):
        self.epochs = params["epochs"]
        self.steps  = params["steps"]

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

    def on_epoch_end(self, epoch, logs=None):
        disc_dict = self.model.discriminator
        gen_dict  = self.model.generator

        d_learning = disc_dict["learning_rate"]
        g_learning = gen_dict["learning_rate"]

        # just useful for learning rate schedulers
        print(f"Learning rate discriminator at epoch {epoch+1} = {d_learning}")
        print(f"Learning rate generator at epoch {epoch+1} = {g_learning}")