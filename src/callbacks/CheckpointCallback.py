from keras import callbacks

class CheckpointCallback(callbacks.Callback):

    def __init__(self, checkpoint_on_epoch=30):
        super().__init__()
        self.checkpoint_on_epoch = checkpoint_on_epoch
        self._checkpoint = 0

    def on_epoch_end(self, epoch, logs=None):
        if(epoch % self.checkpoint_on_epoch == 0):
            self.model.checkpoint(self._checkpoint)
            self._checkpoint += 1