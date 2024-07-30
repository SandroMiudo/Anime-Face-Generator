from tensorflow import data
from tensorflow import io
from tensorflow import dtypes
from tensorflow import image
from tensorflow import random
from ..plot import ImagePlotter as img_plt

class ImageProvider:

    _IMG_HEIGHT = 128
    _IMG_WIDTH  = 128

    @staticmethod
    def normalize_image_zero_to_one(img):
        return img / 255
    
    @staticmethod
    def normalize_image_minus_one_to_one(img):
        return (img - 127.5) / 127.5

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
        ImageProvider._IMG_HEIGHT = height

    @staticmethod
    def _set_width(width:int):
        ImageProvider._IMG_WIDTH = width

    @classmethod
    def build_from_dataset(cls, batch_size, dataset):
        return cls(batch_size=batch_size, dataset=dataset)

    def __init__(self, batch_size=32, should_cache=False, shuffle_buffer=15000, 
                 dataset=None, debug_mode=False, no_augment=False) -> None:
        if(dataset != None):
            self.dataset = dataset
            self.batch_size = batch_size
            return

        if(batch_size < 8):
            Warning("Batch size shouldn't be set to low.")
        if(batch_size > 128 or batch_size < 2):
            print(f"Batch size has to be in range : [2-128], but received {batch_size}")
        dataset = data.Dataset.list_files("dataset/*")
        self.dataset = dataset
        self.batch_size = batch_size
        self.debug_mode = debug_mode
        print(f"{dataset.cardinality()} Images found --\n")
        print("Preprocess data : order --\n")
        print("Converting images to tensors --\n")

        def convert_to_tensor_image(dataset_file_path):
            img          = io.read_file(dataset_file_path)
            tensor_image = io.decode_jpeg(img, channels=3)
            return image.resize(tensor_image, [ImageProvider._IMG_HEIGHT, ImageProvider._IMG_WIDTH])

        self.dataset = self.dataset.map(convert_to_tensor_image)
        print(f"Normalizing images using {ImageProvider.normalize_function_used} --\n")
        self.dataset = self.dataset.map(lambda x : ImageProvider.normalize_function_used(x))
        if not no_augment:
            print(f"Augmenting dataset --\n")
            self.augment()
            print(f"{self.dataset.cardinality()} Images after augmentation --\n")
        print(f"Shuffling images using a buffer size of {shuffle_buffer} -- random order\n")
        self.dataset = self.dataset.shuffle(shuffle_buffer)
        print(f"Creating batches of size {batch_size} --\n")
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)
        if(should_cache):
            print("Caching images --\n")
            dataset = dataset.cache()
        print("Preprocessing data finished --\n")

    def provide_images(self) -> tuple[data.Dataset, int]:
        return self.dataset, self.batch_size
    
    def provide_image_dim(self) -> tuple[int, int, int]:
        return (ImageProvider._IMG_HEIGHT, ImageProvider._IMG_WIDTH, 3)

    def augment(self):
        d1 = self.dataset.map(lambda x : image.random_brightness(x, 0.2))
        d2 = self.dataset.map(lambda x : image.random_contrast(x, 0.5, 1.5))
        d3 = self.dataset.map(lambda x : image.random_saturation(x, 0.5, 1.5))
        #d4 = self.dataset.map(lambda x : image.random_jpeg_quality(x, 95, 98))
        # d5 = self.dataset.map(lambda x : image.random_hue(x, 0.05))

        if(self.debug_mode):
            image_plotter = img_plt.ImagePlotter(self, (5, 2))
            image_plotter.plot_from_datasets([d1, d2, d3])

        self.dataset = self.dataset.concatenate(d1)
        self.dataset = self.dataset.concatenate(d2)
        self.dataset = self.dataset.concatenate(d3)
        # self.dataset = self.dataset.concatenate(d4)
        # self.dataset = self.dataset.concatenate(d5)

    def sample_images(self, skip, take):
        total_batches = len(self.dataset)
        tmp_dataset = self.dataset.skip(skip % total_batches).take(1)
        iterator = iter(tmp_dataset)
        batch = iterator.get_next().numpy()
        random_tensor = random.uniform([take % self.batch_size], 0, self.batch_size, dtype=dtypes.int64)
        return batch[random_tensor.numpy(), ...], take % self.batch_size