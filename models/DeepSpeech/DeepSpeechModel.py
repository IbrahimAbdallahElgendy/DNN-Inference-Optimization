# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import LSTM
from keras.initializers import random_normal
from keras.models import Model
from keras.layers import Dense, Bidirectional, Lambda, Input
from keras.layers import TimeDistributed
import keras.backend as K

class DeepSpeechNet():
    def __init__(self):
        self.net = self.build_network()

    # Define CTC loss
    def ctc_lambda_func(self,args):
        y_pred, labels, input_length, label_length = args

        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build_network(self,input_dim=26, fc_size=2028, rnn_size=2048,
                      output_dim=29,is_train=False):

        input_data = Input(shape=(None, input_dim))  # >>(?, 778, 26)
        init = random_normal(stddev=0.046875)

        x = TimeDistributed(
            Dense(fc_size, name='fc1', kernel_initializer=init,
                  bias_initializer=init, activation='relu'))(input_data)  # >>(?, 778, 2048) #原实现使用clipped_relu激活
        x = TimeDistributed(
            Dense(fc_size, name='fc2', kernel_initializer=init,
                  bias_initializer=init, activation='relu'))(x)  # >>(?, 778, 2048)  #原实现使用clipped_relu激活
        x = TimeDistributed(
            Dense(fc_size, name='fc3', kernel_initializer=init,
                  bias_initializer=init, activation='relu'))(x)  # >>(?, 778, 2048)  #原实现使用clipped_relu激活

        '''Layer 4 BiDirectional RNN - note coreml only supports LSTM BIDIR'''
        x = Bidirectional(LSTM(rnn_size, return_sequences=True,activation='relu',
                               kernel_initializer='glorot_uniform', name='birnn'),
                          merge_mode='sum')(x)
        '''Layer 5+6 Time Dist Layer & Softmax'''
        y_pred = TimeDistributed(
            Dense(output_dim, kernel_initializer=init,bias_initializer=init,
                  activation="softmax"),name="predictions")(x)

        if is_train:
            # Input of labels and other CTC requirements
            labels = Input(name='the_labels', shape=[None, ], dtype='int32')
            input_length = Input(name='input_length', shape=[1], dtype='int32')
            label_length = Input(name='label_length', shape=[1], dtype='int32')


            loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,labels,
                                                                                    input_length,label_length])

            model = Model(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])
        else:
            # Create model.
            model = Model(input_data, y_pred, name='DeepSpeech')

        return model






