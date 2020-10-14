# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding
'''
https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

**References**
- [Sequence to Sequence Learning with Neural Networks
   ](https://arxiv.org/abs/1409.3215)
   不同的句子有不同的长度。大多数句子很短（例如，长度为20-30）
   使用了深LSTM，该LSTM具有4层，每层具有1000个单元，并且具有1000维单词嵌入，
   输入词汇量为160,000，输出词汇量为80,000。
- [Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    ](https://arxiv.org/abs/1406.1078)
'''
class Seq2SeqNet():
    def __init__(self):
        # encoder词典大小
        self.num_encoder_tokens = 10000
        # decoder词典大小
        self.num_decoder_tokens = 10000
        self.latent_dim = 256  # 编码空间的潜在(Latent)维数
        self.net = self.build_word_network(self.num_encoder_tokens,
                                           self.num_decoder_tokens,self.latent_dim)

    def build_network(self,num_encoder_tokens,num_decoder_tokens,latent_dim):
        """ Encoder """
        # 定义一个输入序列并进行处理.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # 我们抛弃`encoder_outputs`，只保留状态
        encoder_states = [state_h, state_c]

        """Decoder"""
        # 使用`encoder_states`作为初始状态来设置解码器
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # 我们将解码器设置为返回完整的输出序列，并返回内部状态。
        # 我们不在训练模型中使用返回状态，但会在推理中使用它们。
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(decoder_outputs)

        """model"""
        # 定义将`encoder_input_data` & `decoder_input_data` 转换为`decoder_target_data`的模型
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_word_network(self,num_encoder_tokens,num_decoder_tokens,
                           latent_dim,lstm_layers=4):
        """ Encoder """
        # 定义一个输入序列并进行处理.
        encoder_inputs = Input(shape=(None,))
        x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
        for i in range(lstm_layers-1):
            x = LSTM(latent_dim,return_sequences=True)(x)
        x, state_h, state_c = LSTM(latent_dim,return_state=True)(x)
        encoder_states = [state_h, state_c]

        """Decoder"""
        # 使用`encoder_states`作为初始状态来设置解码器
        decoder_inputs = Input(shape=(None,))
        x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
        x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
        for i in range(lstm_layers-1):
            x = LSTM(latent_dim, return_sequences=True)(x)
        decoder_outputs = Dense(num_decoder_tokens, activation='softmax',name='predictions')(x)

        """model"""
        # 定义将`encoder_input_data` & `decoder_input_data` 转换为`decoder_target_data`的模型
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # Compile & run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        return model



