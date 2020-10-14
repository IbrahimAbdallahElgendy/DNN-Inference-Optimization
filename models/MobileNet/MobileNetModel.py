# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D,Activation,Reshape,Dropout,GlobalAveragePooling2D
from keras.layers import ZeroPadding2D,BatchNormalization,DepthwiseConv2D,ThresholdedReLU
import keras.backend as K

class MobileNet():
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.classes = 1000
        self.net = self.build_network()

    # def relu6(self, x):
    #     # return K.relu(x, max_value=6)

    def _conv_block(self,inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
        """Adds an initial convolution layer (with batch normalization and relu6).

        # Arguments
            inputs: Input tensor of shape `(rows, cols, 3)`
                (with `channels_last` data format) or
                (3, rows, cols) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(224, 224, 3)` would be one valid value.
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.

        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        filters = int(filters * alpha)
        x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
        x = Conv2D(filters, kernel,
                   padding='valid',
                   use_bias=False,
                   strides=strides,
                   name='conv1')(x)
        x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
        return ThresholdedReLU(0)(x) # 原实现relu6

    def _depthwise_conv_block(self,inputs, pointwise_conv_filters, alpha,
                              depth_multiplier=1, strides=(1, 1), block_id=1):
        """Adds a depthwise convolution block.

        A depthwise convolution block consists of a depthwise conv,
        batch normalization, relu6, pointwise convolution,
        batch normalization and relu6 activation.

        # Arguments
            inputs: Input tensor of shape `(rows, cols, channels)`
                (with `channels_last` data format) or
                (channels, rows, cols) (with `channels_first` data format).
            pointwise_conv_filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the pointwise convolution).
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            depth_multiplier: The number of depthwise convolution output channels
                for each input channel.
                The total number of depthwise convolution output
                channels will be equal to `filters_in * depth_multiplier`.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            block_id: Integer, a unique identification designating the block number.

        # Input shape
            4D tensor with shape:
            `(batch, channels, rows, cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, rows, cols, channels)` if data_format='channels_last'.

        # Output shape
            4D tensor with shape:
            `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
            or 4D tensor with shape:
            `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
            `rows` and `cols` values might have changed due to stride.

        # Returns
            Output tensor of block.
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        x = ZeroPadding2D(padding=(1, 1), name='conv_pad_%d' % block_id)(inputs)
        x = DepthwiseConv2D((3, 3),
                            padding='valid',
                            depth_multiplier=depth_multiplier,
                            strides=strides,
                            use_bias=False,
                            name='conv_dw_%d' % block_id)(x)
        x = BatchNormalization(
            axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
        x = ThresholdedReLU(0)(x)  # 原实现relu6

        x = Conv2D(pointwise_conv_filters, (1, 1),
                   padding='same',
                   use_bias=False,
                   strides=(1, 1),
                   name='conv_pw_%d' % block_id)(x)
        x = BatchNormalization(
            axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
        return ThresholdedReLU(0)(x)  # 原实现relu6

    def build_network(self):

        img_input = Input(shape=self.input_shape)
        alpha = 1.0
        depth_multiplier = 1
        dropout = 1e-3

        x = self._conv_block(img_input, 32, alpha, strides=(2, 2))
        x = self._depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

        x = self._depthwise_conv_block(x, 128, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=2)
        x = self._depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

        x = self._depthwise_conv_block(x, 256, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=4)
        x = self._depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=6)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        x = self._depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

        x = self._depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                                  strides=(2, 2), block_id=12)
        x = self._depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

        # Classification block
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(self.classes, (1, 1),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((self.classes,), name='reshape_2')(x)

        # Create model.
        model = Model(img_input, x, name='MobileNet')

        return model



