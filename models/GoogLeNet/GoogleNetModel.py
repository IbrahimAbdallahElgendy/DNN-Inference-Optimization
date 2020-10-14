# coding=utf-8
# from keras.layers import *
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.models import Model
from keras.regularizers import l2
# import keras.backend as K

class GoogLeNetModel():
    def __init__(self):
        # self.input_shape = (224, 224, 3)
        self.input_shape = (32, 32, 3)
        # self.input_shape = (112, 112, 3)
        self.classes = 1000
        # 搭建模型结构
        self.net = self.create_model(input_shape=self.input_shape, classes=self.classes)

    """定义GoogleNet中的Inception module"""

    # Inception module - main building block
    def inception_module(self, X, filter_sizes):
        # 1x1 covolution
        conv_1x1 = Conv2D(filter_sizes[0], kernel_size=1, strides=1, padding='same', activation='relu',
                          kernel_regularizer=l2(0.0002))(X)

        # Bottleneck layer and 3x3 convolution
        conv_3x3 = Conv2D(filter_sizes[1], kernel_size=1, strides=1, padding='same', activation='relu',
                          kernel_regularizer=l2(0.0002))(X)
        conv_3x3 = Conv2D(filter_sizes[2], kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_regularizer=l2(0.0002))(conv_3x3)

        # Bottleneck layer and 5x5 convolution
        conv_5x5 = Conv2D(filter_sizes[3], kernel_size=1, strides=1, padding='same', activation='relu',
                          kernel_regularizer=l2(0.0002))(X)
        conv_5x5 = Conv2D(filter_sizes[4], kernel_size=5, strides=1, padding='same', activation='relu',
                          kernel_regularizer=l2(0.0002))(conv_5x5)

        # Max pooling and bottleneck layer
        max_pool = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)
        max_pool = Conv2D(filter_sizes[5], kernel_size=1, strides=1, padding='same', activation='relu',
                          kernel_regularizer=l2(0.0002))(max_pool)

        # Concatenate all tensors to 1 tensor
        X = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=3)
        # X = K.concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=3)

        return X

    # 辅助分类器-用于中间阶段的预测
    def aux_classifier(self, X):
        # Average pooling, fc, dropout, fc
        X = AveragePooling2D(pool_size=3, strides=2, padding='same')(X)
        X = Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', activation='relu',
                   kernel_regularizer=l2(0.0002))(X)
        X = Flatten()(X)
        X = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(X)
        X = Dropout(0.7)(X)
        X = Dense(self.classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

        return X

    """GoogleNet模型"""

    # 完整的模型
    def create_model(self, input_shape, classes):
        # 定义输入
        X_input = Input(input_shape)

        # Stage 1 - layers before inception modules
        X = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2(0.0002))(X_input)
        X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Conv2D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu',
                   kernel_regularizer=l2(0.0002))(X)
        X = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu',
                   kernel_regularizer=l2(0.0002))(X)
        X = BatchNormalization(axis=3)(X)
        X = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)

        # Stage 2 - 2 inception modules and max pooling
        X = self.inception_module(X, filter_sizes=[64, 96, 128, 16, 32, 32])
        X = self.inception_module(X, filter_sizes=[128, 128, 192, 32, 96, 64])
        X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

        # Stage 3 - 5 inception modules and max pooling
        X = self.inception_module(X, filter_sizes=[192, 96, 208, 16, 48, 64])
        aux_output_1 = self.aux_classifier(X)  # Auxiliary classifier
        X = self.inception_module(X, filter_sizes=[160, 112, 225, 24, 64, 64])
        X = self.inception_module(X, filter_sizes=[128, 128, 256, 24, 64, 64])
        X = self.inception_module(X, filter_sizes=[112, 144, 288, 32, 64, 64])
        aux_output_2 = self.aux_classifier(X)  # Auxiliary classifier
        X = self.inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
        X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

        # Stage 4 - 2 inception modules and average pooling
        X = self.inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
        X = self.inception_module(X, filter_sizes=[384, 192, 384, 48, 128, 128])
        X = AveragePooling2D(pool_size=4, strides=1, padding='valid')(X)

        # Stage 5 - dropout, linear fc, softmax fc
        X = Flatten()(X)
        X = Dropout(0.4)(X)
        X_output = Dense(classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

        # Create model - 结合主分类器和辅助分类器
        model = Model(inputs=X_input, outputs=[X_output, aux_output_1, aux_output_2])

        return model


