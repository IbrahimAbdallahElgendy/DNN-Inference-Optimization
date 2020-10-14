# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D,BatchNormalization,Activation,AveragePooling2D
from keras import backend as K
import keras.layers as layers

class ResNet():
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.classes = 1000
        self.net = self.build_network()

    def conv_block(self,input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def identity_block(self,input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def build_network(self):

        inputs = Input(shape=self.input_shape)

        # if K.image_data_format() == 'channels_last':
        #     bn_axis = 3
        # else:
        #     bn_axis = 1
        bn_axis = 3

        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        x = Flatten()(x)
        x = Dense(self.classes, activation='softmax', name='predictions')(x)

        # Create model.
        model = Model(inputs, x, name='resnet50')

        return model



