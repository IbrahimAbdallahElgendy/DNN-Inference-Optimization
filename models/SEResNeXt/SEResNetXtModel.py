# -*- coding: utf-8 -*-
"""
https://github.com/taki0112/SENet-Tensorflow/blob/master/SE_ResNeXt.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import GlobalAveragePooling2D,BatchNormalization,Activation
from keras.layers import AveragePooling2D,ZeroPadding3D,ZeroPadding2D
from keras import backend as K
from keras.layers import Concatenate,Reshape,Lambda
import numpy as np

class SEResNeXtNet():
    def __init__(self):
        # # image_size = 32, img_channels = 3, class_num = 10 in cifar10
        self.input_shape = (32, 32, 3)
        self.classes = 10
        self.reduction_ratio = 4
        self.cardinality = 8  # how many split ?
        self.blocks = 3  # res_block ! (split + transition)
        self.depth = 64  # out channel
        self.net = self.build_network()

    def first_layer(self, x, scope):
        with K.name_scope(scope):
            x = Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=1,use_bias=False,padding='SAME')(x)
            x = BatchNormalization(axis=3, scale=False)(x)
            x = Activation("relu")(x)
            return x

    def transform_layer(self, x, stride, scope):
        with K.name_scope(scope):
            x = Conv2D(filters=self.depth,
                       kernel_size=(1, 1),
                       strides=1,use_bias=False,padding='SAME')(x)
            x = BatchNormalization(axis=3, scale=False)(x)
            x = Activation("relu")(x)

            x = Conv2D(filters=self.depth,
                       kernel_size=(3, 3),
                       strides=stride,use_bias=False,padding='SAME')(x)
            x = BatchNormalization(axis=3, scale=False)(x)
            x = Activation("relu")(x)
            return x

    def split_layer(self, input_x, stride, layer_name):
        with K.name_scope(layer_name):
            layers_split = list()
            for i in range(self.cardinality):
                splits = self.transform_layer(input_x, stride=stride,
                                              scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenate(axis=3)(layers_split)

    def transition_layer(self, x, out_dim, scope):
        with K.name_scope(scope):
            x = Conv2D(filters=out_dim,
                       kernel_size=(1, 1),
                       strides=1,use_bias=False,padding='SAME')(x)

            x = BatchNormalization(axis=3, scale=False)(x)
            # x = Activation("relu")(x)
            return x

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with K.name_scope(layer_name):
            squeeze = GlobalAveragePooling2D()(input_x)

            excitation = Dense(int(out_dim / ratio), use_bias=False)(squeeze)
            excitation = Activation("relu")(excitation)
            excitation = Dense(out_dim, use_bias=False)(excitation)
            excitation = Activation("sigmoid")(excitation)

            # excitation = K.reshape(excitation, [-1, 1, 1, out_dim])
            excitation = Reshape((1, 1, out_dim))(excitation)

            # scale = input_x * excitation
            scale = Lambda(lambda x: x[0] * x[1])([input_x ,excitation])

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block):
        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])
            print(input_dim,out_dim)

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=self.reduction_ratio,
                                              layer_name='squeeze_layer_' + layer_num + '_' + str(i))

            print('flag',flag)
            if flag is True:
                print('channel',channel)
                pad_input_x = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(input_x)
                # pad_input_x = ZeroPadding3D(padding=(0,0,channel),data_format='channels_last')(pad_input_x)
                # channels_first 填充轴3和轴4
                pad_input_x = ZeroPadding2D(padding=(0,channel),
                                            data_format='channels_first')(pad_input_x)
            else:
                pad_input_x = input_x

            input_x = Lambda(lambda x: x[0] + x[1])([x , pad_input_x])
            input_x = Activation("relu")(input_x)

        return input_x

    def build_network(self):

        inputs = Input(shape=self.input_shape)

        x = self.first_layer(inputs, scope='first_layer')

        x = self.residual_layer(x, out_dim=64, layer_num='1', res_block=self.blocks)
        x = self.residual_layer(x, out_dim=128, layer_num='2', res_block=self.blocks)
        x = self.residual_layer(x, out_dim=256, layer_num='3', res_block=self.blocks)

        # Classification block
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.classes, activation='softmax', name='predictions')(x)

        # Create model.
        model = Model(inputs, x, name='SEResNeXt')

        return model



