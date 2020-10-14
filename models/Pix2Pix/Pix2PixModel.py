# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Concatenate

from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout

class Pix2pixNet():
    def __init__(self):
        self.input_shape = [256, 256, 3]
        self.OUTPUT_CHANNELS = 3
        self.net = self.build_network()

    def upsample(self,filters, size, x,apply_dropout=False):
        x = Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            use_bias=False)(x)

        x = BatchNormalization()(x)

        if apply_dropout:
            x = Dropout(0.5)(x)

        x = LeakyReLU()(x)

        return x

    def downsample(self,filters, size,x, apply_batchnorm=True):

        x = Conv2D(filters, size, strides=2, padding='same',
                                   use_bias=False)(x)

        if apply_batchnorm:
            x = BatchNormalization()(x)

        x = LeakyReLU()(x)

        return x

    def build_network(self):
        inputs = Input(shape=[256, 256, 3])
        down_stack = [
            [64, 4,False],
            [128, 4,True],
            [256, 4,True],
            [512, 4,True],
            [512, 4,True],
            [512, 4,True],
            [512, 4,True],
            [512, 4,True]
        ]
        up_stack = [
            [512, 4,True],
            [512, 4,True],
            [512, 4,True],
            [512, 4,False],
            [256, 4,False],
            [128, 4,False],
            [64, 4,False],
        ]
        last = Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                               strides=2,
                                               padding='same',
                                               activation='tanh')  # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = self.downsample(down[0], down[1], x,apply_batchnorm=down[2])
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = self.upsample(up[0], up[1],x, apply_dropout=up[2])
            x = Concatenate()([x, skip])

        x = last(x)

        return Model(inputs=inputs, outputs=x,name='pix2pix')