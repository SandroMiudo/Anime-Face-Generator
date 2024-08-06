from model import BaseModel as base_model
from utility.dataset import ImageProvider as img_provider
import keras
import argparse
from exceptions.InferenceException import InferenceException
from exceptions.ArgumentBuilderException import ArgumentBuilderException
from collections.abc import Callable
import os
from keras import optimizers
import warnings
from utility.plot import ImagePlotter as img_plt
from tensorflow import saved_model
from numpy import random
from utility.enums.defs import ImageResolution

SAVE_K_IMG = 4

def print_routine_string(routine:str, **kwargs):
    print(routine, end=' : ')
    for v1,v2 in kwargs.items():
        print(v1, v2, sep=" - ", end=' | ')
    print(end='\n')

def inference(ckpt, inference_count, target:list[int, int]):
    if(ckpt <= 0):
        raise InferenceException("checkpoint must be in the range [1,*]")
    if(inference_count <= 0):
        raise InferenceException("inference count must be in the range [1,*]")
    files_present = os.listdir(os.path.join("ckpt", "weights"))
    if(ckpt > len(files_present)):
        raise InferenceException("checkpoint is not present!")
    artifact = saved_model.load(os.path.join("ckpt", "model", "inference",
        f"inf_v_{target[0]}-{target[1]}"))
    noise_vectors = random.normal(0, 1, 
        (inference_count, artifact.signatures["inference"].inputs[0].shape[1]))
    generated_images = artifact.inference(noise_vectors)
    print("Generating images ...")
    for i in range(inference_count):
        image_plotter = img_plt.ImagePlotter(None, (1, 1))
        image_plotter.plot_from_image(generated_images[i])
        if i < SAVE_K_IMG:
            image_plotter.save(os.path.join("media", f"generation", f"inf-{i+1}-target-{target[0]}.jpg"))

def fit(model:base_model.BaseModel, dataset, epochs, generate_images_per_epoch) -> None:
    history = model.fit(dataset, generate_images_per_epoch, epochs)
    print("training completed :",
          f"generator loss = {history.history["gen_loss"]}",
          f"discriminator loss = {history.history["disc_loss"]}", sep='\n')
    _eval_dict = model.evaluate(quantity=10000)
    print(f"evaluating model : \n\tgenerator loss = {_eval_dict["gen_loss"]}", 
          f"\tdiscriminator loss = {_eval_dict["disc_loss"]}",
          f"\tgenerator accuracy = {_eval_dict["gen_acc"]}",
          f"\tdiscriminator accuracy = {_eval_dict["disc_acc"]}",
          sep='\n')
    model.export()

def train(g_learning_rate : float | optimizers.schedules.LearningRateSchedule, 
          d_learning_rate : float | optimizers.schedules.LearningRateSchedule, 
          batch_size, generate_images_per_epoch, noise_vector, epochs,
          no_augment, target: list[int, int]):
    if(isinstance(g_learning_rate, float) and 
       (g_learning_rate <= 0 or d_learning_rate <= 0)):
        raise Exception(
            "learning rate (generator and discriminator) have to be greater than zero")
    print_routine_string("train", generator_learning_rate=g_learning_rate,
        discriminator_learning_rate=d_learning_rate, batch_size=batch_size, 
        generated_images_per_epoch=generate_images_per_epoch, noise_vector=noise_vector,
        epochs=epochs)
    
    img_provider.ImageProvider._set_height(target[0])
    img_provider.ImageProvider._set_width(target[1])
    provider = img_provider.ImageProvider(batch_size=batch_size, no_augment=no_augment)
    dataset, batch_size = provider.provide_images()
    input_shape = provider.provide_image_dim()
    b_model  =  base_model.BaseModel(g_learning_rate, d_learning_rate, noise_vector, batch_size,
                    input_shape)

    fit(b_model, dataset, epochs, generate_images_per_epoch)

def load(generate_images_per_epoch, epochs, no_augment, target:list[int, int]):
    model_file = os.path.join("ckpt", "model", f"model_tgt_{target[0]}-{target[1]}.keras")
    model = keras.models.load_model(model_file)
    batch_size = model.batch_size
    img_provider.ImageProvider._set_height(target[0])
    img_provider.ImageProvider._set_width(target[1])
    provider = img_provider.ImageProvider(batch_size, no_augment=no_augment)
    dataset, _ = provider.provide_images()
    fit(model, dataset, epochs, generate_images_per_epoch)
    
class ArgumentBuilder:
    inference_callable : Callable[[int, int, list[int, int]], None]=None
    train_callable     : Callable[[float | optimizers.schedules.LearningRateSchedule, 
                                   float | optimizers.schedules.LearningRateSchedule, 
                                   int, int, int, int, bool, list[int, int]], None]=None
    load_callable      : Callable[[int, int, bool, list[int, int]], None]=None

    def __init__(self, arguments):
        self.variables = arguments

    @staticmethod
    def build(var_dict):
        argument_builder : ArgumentBuilder = ArgumentBuilder(var_dict) 
        if(var_dict["trainable"]):
            argument_builder.train_callable = train
        elif(var_dict["load_model"]):
            argument_builder.load_callable = load
        elif(var_dict["inference"]):
            argument_builder.inference_callable = inference
        else:
            raise ArgumentBuilderException(
                "base argument has to be provided!")

        if not argument_builder.valid_resolutions():
            raise ArgumentBuilderException(
                "target resolution not supported!")

        if argument_builder.valid_learning_arguments("learning_rate", [1,2],
            "1 or 2 learning rates expected, but received %d arguments."):
            argument_builder.build_components_learning_rates(
                argument_builder.variables["learning_rate"][0],
                argument_builder.variables["learning_rate"][1])

        if argument_builder.valid_learning_arguments("exp_decay", [3,6],
            "Expected 3 or 6 arguments for the exp decay learning rate schedule, "
                "but received %d arguments."):
            _exp_list = argument_builder.variables["exp_decay"]
            argument_builder.build_components_learning_rates(
                optimizers.schedules.ExponentialDecay(_exp_list[0], _exp_list[1],
                _exp_list[2]),
                optimizers.schedules.ExponentialDecay(_exp_list[3], _exp_list[4],
                _exp_list[5]))

        if argument_builder.valid_learning_arguments("poly_decay", [4,8],
            "Expected 4 or 8 arguments for the poly decay learning rate schedule, "
                "but received %d arguments."):
            _poly_list = argument_builder.variables["poly_decay"]
            argument_builder.build_components_learning_rates(
                optimizers.schedules.PolynomialDecay(_poly_list[0], _poly_list[1],
                _poly_list[2], _poly_list[3]),
                optimizers.schedules.PolynomialDecay(_poly_list[4], _poly_list[5],
                _poly_list[6], _poly_list[7]))
        
        if argument_builder.valid_learning_arguments_const():
            _const_bounds_list = argument_builder.variables["const_bounds"]
            _const_values_list = argument_builder.variables["const_values"]
            argument_builder.build_components_learning_rates(
                optimizers.schedules.PiecewiseConstantDecay(
                    _const_bounds_list[:len(_const_bounds_list)//2],
                    _const_values_list[:len(_const_values_list)//2]),
                optimizers.schedules.PiecewiseConstantDecay(
                    _const_bounds_list[len(_const_bounds_list)//2:],
                    _const_values_list[len(_const_values_list)//2:]))

        if argument_builder.variables["trainable"] and(
           not(hasattr(argument_builder, "_generator")) or \
           not(hasattr(argument_builder, "_discriminator"))):
            raise ArgumentBuilderException(
                "Learning Rate or Learning Rate Schedule is required.")

        return argument_builder

    def valid_learning_arguments(self, search_entry : str, search_values: list[int, int],
        miss_search_str : str):
        if(self.variables[search_entry] != None and 
           len(self.variables[search_entry]) not in search_values):
            raise ArgumentBuilderException(
                miss_search_str % len(self.variables[search_entry]))
        elif(self.variables[search_entry] != None):
            if len(self.variables[search_entry]) == search_values[0]:
                self.variables[search_entry] = self.variables[search_entry]*2

            return True
        return False # indicates that this option was not specified
        
    def valid_learning_arguments_const(self):
        search_value = [1,2]
        if self.variables["constant_decay"] != None and \
           self.variables["constant_decay"] in search_value:
            if self.variables["const_bounds"] == None or \
               self.variables["const_values"] == None:
               raise ArgumentBuilderException(
                   "Constant decay learning rate options consist of three subparts,"
                   " but not all subparts were specified.")
            if self.in_range("const_bounds", "const_values", 0): 
                raise ArgumentBuilderException(
                    "Specified more than allowed boundary arguments")
            if self.variables["constant_decay"] == search_value[0] and \
                self.in_range("const_values", "const_bounds", 1) or \
                self.variables["constant_decay"] == search_value[1] and \
                self.in_range("const_values", "const_bounds", 2):
                raise ArgumentBuilderException(
                    "Specified more than allowed value arguments")
            if self.variables["constant_decay"] == search_value[1] and \
                (len(self.variables["const_bounds"]) % 2 != 0 or \
                 len(self.variables["const_values"]) % 2 != 0):
                raise ArgumentBuilderException(
                    "Only unit changes allowed.")
            if self.variables["constant_decay"] == search_value[0]:
                self.variables["const_bounds"] = self.variables["const_bounds"]*2
                self.variables["const_values"] = self.variables["const_values"]*2
            return True
        return False

    def valid_resolutions(self): # currently not using 32x32
        self.variables["target"] = self.variables["target"] * 2
        _tgt = self.variables["target"]
        if _tgt == ImageResolution.x_64_64.value or _tgt == ImageResolution.x_128_128.value \
            or _tgt == ImageResolution.x_256_256.value or _tgt == ImageResolution.x_512_512.value \
            or _tgt == ImageResolution.x_48_48.value or _tgt == ImageResolution.x_96_96.value \
            or _tgt == ImageResolution.x_56_56.value or _tgt == ImageResolution.x_112_112.value: 
            return True
        
        return False

    def in_range(self, search_entry : str, search_entry_2 : str, i : int):
        return (len(self.variables[search_entry]) - i) > len(self.variables[search_entry_2]) 

    def build_components_learning_rates(
            self, 
            l_generator : float | optimizers.schedules.LearningRateSchedule, 
            l_discriminator : float | optimizers.schedules.LearningRateSchedule):
        self._generator = l_generator
        self._discriminator = l_discriminator

    def run(self):
        if(self.variables == None):
            raise ArgumentBuilderException(
                "build has to be called before running!")

        if(self.variables["learning_rate"] != None and 
            (self.variables["exp_decay"] != None or self.variables["poly_decay"] != None)):
            warnings.warn("Learning rate schedule was specified in conjuction with float type learning rate.\n"
                    "Float type learning rate is used in the optimizer.\nIf this was not intended, \n"
                    "make sure to leave out the floating type learning rate.")

        elif(self.variables["exp_decay"] != None and self.variables["poly_decay"] != None):
            warnings.warn("Both learning rate schedules were specified.\nUsing exponential decay schedule.\n"
                    "If this was not intended, make sure to leave out one learing rate schedule.")

        elif(self.variables["exp_decay"] != None or self.variables["poly_decay"] != None):
            warnings.warn("It is required that the arguments to the specified learning rate schedule, contain the correct format.\n"
                    "If this is not the case, this might lead to unexpected behaviour.")

        if(self.train_callable != None):
            self.train_callable(self._generator, self._discriminator,
                self.variables["batch_size"], self.variables["generate_per_epoch"],
                self.variables["noise_vector"], self.variables["epochs"], 
                self.variables["no_augment"], self.variables["target"])
        elif(self.load_callable != None):
            self.load_callable(self.variables["generate_per_epoch"],
                self.variables["epochs"], self.variables["no_augment"],
                self.variables["target"])
        elif(self.inference_callable != None):
            self.inference_callable(self.variables["checkpoint"], 
                self.variables["inference_count"], self.variables["target"])
    
def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("-l", "--load-model", help='load model (arch, weights, state)', 
        dest="load_model", action='store_true')
    argument_parser.add_argument("--version", action='version', version='1.0')
    argument_parser.add_argument("-t", "--train", help='specifies if train should occur',
        action='store_true', dest='trainable')
    argument_parser.add_argument("--learning-rate", nargs='+', dest="learning_rate", type=float,
        help="If one argument is passed in, the learning rate is used for both models.\n"
             "Otherwise (meaning 2 arguments) the first learning rate is used for the generator and" 
             "the second learning rate for the discriminator.")
    argument_parser.add_argument("--inference", dest="inference", action="store_true")
    argument_parser.add_argument("--inference-count", dest="inference_count", type=int, default=1)
    argument_parser.add_argument("--ckpt", dest="checkpoint", type=int, default=1)
    argument_parser.add_argument("--noise-vector", dest="noise_vector", type=int, default=16)
    argument_parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    argument_parser.add_argument("--generate-per-epoch", dest="generate_per_epoch", type=int, default=1)
    argument_parser.add_argument("--epochs", dest="epochs", type=int, default=100)
    argument_parser.add_argument("--exponential-decay", nargs= '+', type=float, help="f1 = initial learning rate (0.01)\n"
        "f2 = decay steps (100000)\nf3 = decay rate (0.96)\nIf 3 arguments are passed, these options are"
        "used for both models.\n"
        "If 6 arguments are passed, f1-f3 are used for the generator and f4-f7 for the discriminator.", 
        dest="exp_decay")
    argument_parser.add_argument("--polynomial-decay", nargs='+', type=float, help="f1 = initial learning rate (0.01)\n"
        "f2 = decay steps (100000)\nf3 = end learning rate (0.0001)\nf4 = power (0.5)", dest="poly_decay")
    argument_parser.add_argument("--constant-decay", dest="constant_decay", type=int,
        help="This learning rate schedule behaves different, than the previous two options.\n"
        "The option for the constant decay learning rate schedule is subdivided into three parts."
        "Each of that subpart is required and can't be left empty!\n"
        "This part is just used to indicate that the constant decay learning rate schedule should be used.\n"
        "Parts that should following are : '--constant-decay-boundaries' and '--constant-decay-values'\n."
        "Example usage : --constant-decay --constant-decay-boundaries 100000 110000 "
        "--constant-decay-values 1.0 0.5 0.1")
    argument_parser.add_argument("--constant-decay-boundaries", nargs="+", type=int, dest="const_bounds",
        help="Boundaries at which the scheduler, will set a new learning rate.\n"
             "These arguments should be in non decreasing order and should match exactly (or len - 1)"
             "the arguments specified in '--constant-decay-values'.")
    argument_parser.add_argument("--constant-decay-values", nargs="+", type=float, dest="const_values",
        help="Values at which the scheduler will adjust its learning rate at a given boundary X."
             "These arguments should be in non decreasing order and the arguments length must "
             "match exactly (or len + 1) the arguments provided by '--constant-decay-boundaries'.")
    argument_parser.add_argument("--no-augment", action="store_true", dest="no_augment", 
        help="Specifies if data augmentation should not be applied to dataset : defaults to false")
    argument_parser.add_argument("--target", nargs=1, help="Defines target spatial dimension"
                                 "Avaiblable targets : 64, 128, 256, 512",
        default=[64], type=int, dest="target")
    namespace_obj = argument_parser.parse_args()
    var_dict      = vars(namespace_obj)

    ArgumentBuilder.build(var_dict).run()

if __name__ == "__main__":
    parse_arguments()