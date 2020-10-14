# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.layers import Dense,Lambda
from keras.layers import Input
from keras.losses import binary_crossentropy
import keras.backend as K

class VAENet():
    def __init__(self):
        self.image_size = 32
        self.classes = 1000
        self.net = self.build_network(self.image_size)

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(self,args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def init_network(self,image_size):
        # network parameters
        original_dim = image_size * image_size
        input_shape = (original_dim,)
        intermediate_dim = 512
        latent_dim = 2

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # 使用reparameterization技巧将采样作为输入项推出，请注意，TensorFlow backend不需要"output_shape"
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid',name='predictions')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')
        vae.summary()

        vae.compile(optimizer='adam',loss=binary_crossentropy(inputs,outputs))

        return encoder,decoder,vae

    def build_network(self, image_size):
        # network parameters
        original_dim = image_size * image_size
        input_shape = (original_dim,)
        intermediate_dim = 512
        latent_dim = 2

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # 使用reparameterization技巧将采样作为输入项推出，请注意，TensorFlow backend不需要"output_shape"
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        '''
        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        '''

        # build decoder model
        '''
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        '''
        x = Dense(intermediate_dim, activation='relu')(z)
        outputs = Dense(original_dim, activation='sigmoid', name='predictions')(x)

        '''
        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        '''

        # instantiate VAE model
        '''outputs = decoder(encoder(inputs)[2])'''
        vae = Model(inputs, outputs, name='vae_mlp')

        return vae



