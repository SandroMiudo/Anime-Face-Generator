import keras.saving
from model import BaseModel as base_model, DiscriminatorModel as discriminator_model, \
    GeneratorModel as gen_model
from utility.dataset import ImageProvider as img_provider
from keras import optimizers
from keras import metrics
from keras import losses
import keras
import argparse
from exceptions import InferenceException, ArgumentBuilderException
from collections.abc import Callable
from keras.callbacks import History
import os

def print_routine_string(routine:str, **kwargs):
    print(routine, end=' : ')
    for v1,v2 in kwargs.items():
        print(v1, v2, sep=" - ", end=' | ')
    print(end='\n')

def inference(ckpt, inference_count):
    if(ckpt <= 0):
        raise InferenceException("checkpoint must be in the range [1,*]")
    if(inference_count <= 0):
        raise InferenceException("inference count must be in the range [1,*]")
    files_present = os.listdir(os.path.join("ckpt", "weights"))
    if(ckpt > files_present):
        raise InferenceException("checkpoint is not present!")
    weights_file_path = os.path.join("ckpt", "weights", f"ckpt_{ckpt-1}.weights.h5")
    model = keras.models.load_model(weights_file_path)
    for i in range(inference_count):
        generated_image = model.predict(inference_count)

def fit(model:base_model.BaseModel, dataset, epochs, generate_images_per_epoch) -> None:
    history = model.fit(dataset, generate_images_per_epoch, epochs)

def train(g_learning_rate, d_learning_rate, batch_size, generate_images_per_epoch,
    noise_vector, epochs):
    if(g_learning_rate <= 0 or d_learning_rate <= 0):
        raise Exception("learning rate (generator and discriminator) have to be greater than zero")
    print_routine_string("train", generator_learning_rate=g_learning_rate,
        discriminator_learning_rate=d_learning_rate, batch_size=batch_size, 
        generated_images_per_epoch=generate_images_per_epoch, noise_vector=noise_vector,
        epochs=epochs)
    provider = img_provider.ImageProvider(batch_size=batch_size)
    dataset, batch_size = provider.provide_images()
    input_shape = provider.provide_image_dim()
    b_model  =  base_model.BaseModel(g_learning_rate, d_learning_rate, noise_vector, batch_size,
                    input_shape)

    fit(b_model, dataset, epochs, generate_images_per_epoch)

def load(generate_images_per_epoch, epochs):
    model_file = os.path.join("ckpt", "model", "model.keras")
    model = keras.models.load_model(model_file)
    provider = img_provider.ImageProvider()
    dataset, _ = provider.provide_images()
    fit(model, dataset, epochs, generate_images_per_epoch)
    
class ArgumentBuilder:
    inference_callable : Callable[[int, int], None]=None
    train_callable     : Callable[[float, float, int, int, int, int], History]=None
    load_callable      : Callable[[int, int], History]=None

    def build(self, var_dict):
        if("trainable" in var_dict):
            self.train_callable = train
        elif("load_model" in var_dict):
            self.load_callable = load
        elif("inference" in var_dict):
            self.inference_callable = inference
        else:
            raise ArgumentBuilderException("base argument has to be provided!")
        self.variables = var_dict
        return self

    def run(self):
        if(self.variables == None):
            raise ArgumentBuilderException("build has to be called before running!")

        if(self.train_callable != None):
            self.train_callable(self.variables["g_learning"], self.variables["d_learning"],
                self.variables["batch_size"], self.variables["generate_per_epoch"],
                self.variables["noise_vector"], self.variables["epochs"])
        elif(self.load_callable != None):
            self.load_callable(self.variables["generate_per_epoch"],
                self.variables["epochs"])
        elif(self.inference_callable != None):
            self.inference_callable(self.variables["checkpoint"], 
                self.variables["inference_count"])
    
def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-l", "--load-model", help='load model (arch, weights, state)', 
        dest="load_model", action='store_true')
    argument_parser.add_argument("--version", action='version', version='1.0')
    argument_parser.add_argument("-t", "--train", help='specifies if train should occur',
        action='store_true', dest='trainable')
    argument_parser.add_argument("--g-learning-rate", dest="g_learning", type=float, default=1e-4)
    argument_parser.add_argument("--d-learning-rate", dest="d_learning", type=float, default=1e-4)
    argument_parser.add_argument("--inference", dest="inference", action="store_true")
    argument_parser.add_argument("--inference-count", dest="inference_count", type=int, default=1)
    argument_parser.add_argument("--ckpt", dest="checkpoint", type=int, default=1)
    argument_parser.add_argument("--noise_vector", dest="noise_vector", type=int, default=16)
    argument_parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    argument_parser.add_argument("--generate-per-epoch", dest="generate_per_epoch", type=int, default=1)
    argument_parser.add_argument("--epochs", dest="epochs", type=int, default=100)

    namespace_obj = argument_parser.parse_args()
    var_dict      = vars(namespace_obj)

    ArgumentBuilder().build(var_dict).run()

if __name__ == "__main__":
    parse_arguments()