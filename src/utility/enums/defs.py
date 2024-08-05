from enum import Enum
import numpy as np

class ImageResolution(Enum):
    x_64_64   = [64, 64] 
    x_128_128 = [128, 128]
    x_256_256 = [256, 256]
    x_512_512 = [512, 512]
    x_48_48   = [48, 48]
    x_96_96   = [96, 96]
    x_56_56   = [56, 56]
    x_112_112 = [112, 112]

type ResSequence = list[int]

res_seq_1 : ResSequence = [ImageResolution.x_64_64.value[0], 
             ImageResolution.x_128_128.value[0],
             ImageResolution.x_256_256.value[0],
             ImageResolution.x_512_512.value[0]]

res_seq_2 : ResSequence = [ImageResolution.x_48_48.value[0],
                         ImageResolution.x_96_96.value[0]]

res_seq_3 : ResSequence = [ImageResolution.x_56_56.value[0],
                         ImageResolution.x_112_112.value[0]]

class ImageResolutionOps:
    _TARGET = []

    @staticmethod
    def loc_range(shape:list[int, int]):
        _tgt_seq = []
        if shape[0] in res_seq_1:
            _tgt_seq = res_seq_1
        elif shape[0] in res_seq_2:
            _tgt_seq = res_seq_2
        elif shape[0] in res_seq_3:
            _tgt_seq = res_seq_3

        arr_1 = np.array(ImageResolution.x_64_64.value)
        arr_2 = np.array(ImageResolution.x_128_128.value)
        arr_3 = np.array(ImageResolution.x_256_256.value)
        arr_4 = np.array(ImageResolution.x_512_512.value)
        arr_5 = np.array(ImageResolution.x_48_48.value)
        arr_6 = np.array(ImageResolution.x_96_96.value)
        arr_7 = np.array(ImageResolution.x_56_56.value)
        arr_8 = np.array(ImageResolution.x_112_112.value)

        stacked_arr = np.vstack((arr_1, arr_2, arr_3, arr_4, 
                                 arr_5, arr_6, arr_7, arr_8))
 
        ImageResolutionOps._TARGET = [x[0] in _tgt_seq and x[0] <= shape[0] 
                            for x in stacked_arr]

    @staticmethod
    def in_range_tgt(tgt=int):
        if len(ImageResolutionOps._TARGET) == 0 or \
            len(ImageResolutionOps._TARGET) <= tgt:
            return False

        return ImageResolutionOps._TARGET[tgt]
    
    @staticmethod
    def index(target_res):
        index = -1
        if target_res == ImageResolution.x_64_64:
            index = 0
        elif target_res == ImageResolution.x_128_128:
            index = 1
        elif target_res == ImageResolution.x_256_256:
            index = 2
        elif target_res == ImageResolution.x_512_512:
            index = 3
        elif target_res == ImageResolution.x_48_48:
            index = 4
        elif target_res == ImageResolution.x_96_96:
            index = 5
        elif target_res == ImageResolution.x_56_56:
            index = 6
        elif target_res == ImageResolution.x_112_112:
            index = 7
        return index
    
    @staticmethod
    def tgt_in(target_res):
        if len(ImageResolutionOps._TARGET) == 0:
            return False
        
        return ImageResolutionOps._TARGET[ImageResolutionOps.index(target_res)]