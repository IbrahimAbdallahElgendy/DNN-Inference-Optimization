# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import BatchNormalization, Convolution2D, Dense, LeakyReLU, \
        Input, merge, Reshape, UpSampling2D,Conv2DTranspose
from keras.models import Model

class ChairsNet():
    def __init__(self):
        self.net = self.build_network()

    def build_network(self, class_len=809, view_len=4, transf_param_len=12):
        class_input = Input(shape=(class_len,), name='class')
        view_input = Input(shape=(view_len,), name='view')
        transf_param_input = Input(shape=(transf_param_len,), name='transf_param')

        # Hidden representation for input parameters
        fc1 = LeakyReLU()(Dense(512)(class_input))
        fc1 = LeakyReLU()(Dense(512)(fc1))
        fc2 = LeakyReLU()(Dense(512)(view_input))
        fc2 = LeakyReLU()(Dense(512)(fc2 ))
        fc3 = LeakyReLU()(Dense(512)(transf_param_input))
        fc3 = LeakyReLU()(Dense(512)(fc3))

        x = merge([fc1, fc2, fc3], mode='concat')
        x = LeakyReLU()(Dense(1024)(x))
        x = LeakyReLU()(Dense(1024)(x))


        x = LeakyReLU()(Dense(16384)(x))
        x = Reshape((8,8,-1), name='reshape_1')(x)

        num_kernels = [256,92,48,1]
        for idx in range(3):
            # Upsample input
            x = UpSampling2D((2, 2))(x)
            x = LeakyReLU()(Convolution2D(num_kernels[idx],(4, 4), border_mode='same')(x))

            x = LeakyReLU()(Convolution2D(num_kernels[idx+1],(3, 3),border_mode='same')(x))
            # x = BatchNormalization()(x)

        # Last deconvolution layer: Create 3-channel image.
        x = UpSampling2D((2, 2))(x)
        x = Convolution2D(1, 4, 4, border_mode='same')(x)

        # Compile the model
        model = Model(input=[class_input, transf_param_input, view_input], output=x)
        model.compile(optimizer='adam', loss='msle')

        return model



