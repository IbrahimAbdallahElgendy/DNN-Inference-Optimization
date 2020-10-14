# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense,Activation
from keras.layers import Input
from keras.layers import Conv2D
from keras.optimizers import Adam

class DQNNet():
    def __init__(self):
        self.input_shape=(80, 80, 4)
        self.net = self.build_network()

    def build_network(self):
        print("Now we build the model")
        img_input = Input(shape=self.input_shape)
        #The first hidden layer convolves 32 filters of 8x8 with stride 4 with the input image and applies a rectifier nonlinearity
        x = Conv2D(32, (8, 8), strides=(4, 4), padding='same')(img_input)
        x = Activation('relu')(x)
        # The second hidden layer convolves 64 filters of 4x4 with stride 2, again followed by a rectifier nonlinearity.
        x = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(x)
        x = Activation('relu')(x)
        # Thisisfollowed by a third convolutional layerthat convolves 64 filters of 3x3with stride 1 followed by a rectifier.
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
        x = Activation('relu')(x)
        #The final hidden layer is fully-connected and consists of 512 rectifier units.
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        # The output layer is a fully-connected linear layer with a single output for each valid action.
        # The number of valid actions varied between 4 and 18 on the games we considered.
        x = Dense(18,name='predictions')(x)

        # Create model.
        model = Model(img_input, x, name='DQN')
        LEARNING_RATE = 1e-4
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

        return model