from typing import Any
from keras import layers

class Conv2DBlockBuilder:

    @staticmethod
    def construct(filters:int, kernel_size:tuple[int, int], i_shape:int|None=None) -> list[layers.Layer]:
        block = []
        if i_shape == None:
            block.append(layers.Conv2D(filters, kernel_size, padding='same', activation='relu'))
        else:
            block.append(layers.Conv2D(filters, kernel_size, padding='same', activation='relu', 
                                       input_shape=i_shape))
        block.append(layers.BatchNormalization())
        block.append(layers.Conv2D(filters, kernel_size, padding='same', activation='relu'))
        block.append(layers.BatchNormalization())

        return block
    
class Conv2D_T_BlockBuilder:
    
    @staticmethod
    def construct(filters:int, kernel_size:tuple[int, int]) -> list[layers.Layer]:
        block = []
        block.append(layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2,2),
                    padding='same'))
        block.append(layers.Dropout(0.1))
        block.extend(Conv2DBlockBuilder.construct(filters, kernel_size))

        return block
    
class LayerIterator:

    def __init__(self, layers):
        self._layers = layers

    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self) -> layers.Layer:
        if(len(self._layers) == self._index):
            raise StopIteration()
        x = self._layers[self._index]
        self._index += 1
        return x
    
    def __call__(self, x, training):
        for layer in self:
            x = layer(x, training=training)
        return x