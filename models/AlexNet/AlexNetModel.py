# coding: utf-8
from keras.models import Model,Input,Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,  BatchNormalization

class AlexNet():
    def __init__(self):
        # self.input_shape = (224, 224, 3)
        self.input_shape = (224, 224, 3)
        self.net = self.build_network()

    def build_network(self):
        inputs = Input(shape=self.input_shape)

        x = Conv2D(96, (11, 11), strides=4, activation='relu', padding='valid')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2, padding='valid')(x)

        x = Conv2D(256, (5, 5), strides=1, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2, padding='valid')(x)

        x = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu')(x)
        x = Conv2D(384, (3, 3), strides=1, padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=1, padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2, padding='valid')(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)

        outputs = Dense(1000, activation='softmax',name='predictions')(x)

        model = Model(inputs=[inputs], outputs=[outputs])

        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model