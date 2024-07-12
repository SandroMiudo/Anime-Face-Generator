from tensorflow import data
from tensorflow import io
from tensorflow import dtypes
from tensorflow import image
from tensorflow import random

class ImageProvider:

    IMG_HEIGHT = 128
    IMG_WIDTH  = 128

    @staticmethod
    def normalize_image_zero_to_one(img):
        return img / 255
    
    @staticmethod
    def normalize_image_minus_one_to_one(img):
        return (img - 127.5) / 127.5

    @staticmethod
    def convert_to_tensor_image(dataset_file_path):
        img        = io.read_file(dataset_file_path)
        tensor_image = io.decode_image(img, 3, dtype=dtypes.float16)
        return image.resize(tensor_image, [ImageProvider.IMG_HEIGHT, ImageProvider.IMG_WIDTH])

    @staticmethod
    def scale_image(image): 
        return ImageProvider.normalize_function_used(image)    

    normalize_function_used = normalize_image_zero_to_one

    @staticmethod
    def _set_normalize_function(norm_function):
        if(norm_function != ImageProvider.normalize_image_minus_one_to_one and 
           norm_function != ImageProvider.normalize_image_zero_to_one):
            return
        ImageProvider.normalize_function_used = norm_function

    @staticmethod
    def _set_height(height:int):
        ImageProvider.IMG_HEIGHT = height

    @staticmethod
    def _set_width(width:int):
        ImageProvider.IMG_WIDTH = width

    def __init__(self, batch_size=32, should_cache=False, shuffle_buffer=1000) -> None:
        if(batch_size < 8):
            Warning("Batch size shouldn't be set to low.")
        if(batch_size > 128 or batch_size < 2):
            print(f"Batch size has to be in range : [2-128], but received {batch_size}")
        dataset = data.Dataset.list_files("../../dataset/*")
        self.dataset = dataset
        print(f"{dataset.cardinality()} Images found --\n")
        print("Preprocess data : order --\n")
        print("Converting images to tensors --\n")
        dataset = dataset.map(lambda x : ImageProvider.convert_to_tensor_image(x))
        print(f"Normalizing images using {ImageProvider.normalize_function_used} --\n")
        dataset = dataset.map(lambda x : ImageProvider.normalize_function_used(x))
        print(f"Shuffling images using a buffer size of {shuffle_buffer} -- random order\n")
        dataset = dataset.shuffle(shuffle_buffer)
        print(f"Creating batches of size {batch_size} --\n")
        dataset = dataset.batch(batch_size, drop_remainder=True)
        if(should_cache):
            print("Caching images --\n")
            dataset = dataset.cache()
        print("Preprocessing data finished --\n")

    def provide_images(self):
        return self.dataset

    def sample_images(self, skip, take):
        total_batches = len(self.dataset)
        if(take > total_batches): 
            return None
        tmp_dataset = self.dataset.skip(skip).take(1)
        iterator = iter(tmp_dataset)
        batch = iterator.get_next()
        random_tensor = random.uniform([take], 0, len(batch), dtype=dtypes.int64)
        return batch[random_tensor, ...]