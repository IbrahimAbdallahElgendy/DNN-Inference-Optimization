# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D,BatchNormalization,Activation
from keras import backend as K
import keras

class InceptionV3Net():
    def __init__(self):
        self.input_shape = (299, 299, 3)
        self.classes = 1000
        self.net = self.build_network()

    def conv2d_bn(self,x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1),
                  name=None):
        """
        conv + BN 的功能函数

         # Arguments
             x: 输入张量。
             filters: `Conv2D`中的过滤器
             num_row: 卷积核的高度
             num_col: 卷积核的宽度
             padding: `Conv2D`中的填充模式
             strides: 在Conv2D中的步幅
             name: ops名字; 对于卷积将变为`name + '_conv'`，对于批处理规范层将变为`name + '_bn'`

         # Returns
             应用`Conv2D`和`BatchNormalization`后的输出张量.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    def build_network(self):

        inputs = Input(shape=self.input_shape)

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            #channels_last
            channel_axis = 3

        x = self.conv2d_bn(inputs, 32, 3, 3, strides=(2, 2), padding='valid')
        x = self.conv2d_bn(x, 32, 3, 3, padding='valid')
        x = self.conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding='valid')
        x = self.conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed0')

        # mixed 1: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed1')

        # mixed 2: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = keras.layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = keras.layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, 1, 1)
        branch3x3 = self.conv2d_bn(branch3x3, 320, 3, 3,
                              strides=(2, 2), padding='valid')

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = self.conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = keras.layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = keras.layers.concatenate(
                [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = keras.layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = keras.layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name='mixed' + str(9 + i))

        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(self.classes, activation='softmax', name='predictions')(x)

        # Create model.
        model = Model(inputs, x, name='inception_v3')


        return model

