#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


class EucDist(Layer):
    """
    Keras Custom Layer Implementation for calculating Euclidean Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(EucDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(EucDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
