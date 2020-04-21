#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras as K


class Generator(K.Model):

    def __init__(self, output_channels=3):
        super(Generator, self).__init__()

        # encoder
        filter_sizes = [[64, [4, 4], 2, False],
                        [128, [4, 4], 2, True],
                        [256, [4, 4], 2, True],
                        [512, [4, 4], 2, True],
                        [512, [4, 4], 2, True],
                        [512, [4, 4], 2, True],
                        [512, [4, 4], 2, True],
                        [512, [4, 4], 2, True]]
        self.encoders = \
          [self.conv(batch_norm=bn, filters=fsize,
                     kernel_size=(4, 4), strides=2,
                     padding='same', name=f'conv{i+1}')
           for i, (fsize, ksize, s, bn) in enumerate(filter_sizes)]
        
        dfilter_sizes = [[512, [4, 4], 2, True],
                        [512, [4, 4], 2, True],
                        [512, [4, 4], 2, True],
                        [512, [4, 4], 2, False],
                        [256, [4, 4], 2, False],
                        [128, [4, 4], 2, False],
                        [64, [4, 4], 2, False]]
        
        initializer = tf.random_normal_initializer(0., 0.02)

        self.decoders =\
          [self.deconv(use_dropout=do, filters=fsize, kernel_size=ksize,
                       strides=s, padding='same', use_bias=False,
                       kernel_initializer=initializer,
                       name=f'deconv{i+1}')
           for i, (fsize, ksize, s, do) in enumerate(dfilter_sizes)]

        self.concate = K.layers.Concatenate()
        self.top = K.layers.Conv2DTranspose(
            output_channels, 4, strides=2, padding='same',
            kernel_initializer=initializer, activation='tanh')

    def call(self, inputs, training=False):
        x = inputs
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for decoder, skip in zip(self.decoders, skips):
            x = self.concate([decoder(x), skip])
            # x = K.layers.Concatenate()([decoder(x), skip])
        return self.top(x)

    def conv(self, batch_norm=True, **kwargs):
        slope = kwargs.get('slope', 0.2)
        seq = K.Sequential()
        seq.add(K.layers.Conv2D(**kwargs))
        
        if batch_norm:
            seq.add(K.layers.BatchNormalization())
        seq.add(K.layers.LeakyReLU(slope))
        return seq

    def deconv(self, batch_norm=True, use_dropout=True, **kwargs):
        rate = kwargs.pop('dropout_rate', 0.5)
        seq = K.Sequential()
        seq.add(K.layers.Conv2DTranspose(**kwargs))
        
        if batch_norm:
            seq.add(K.layers.BatchNormalization())
            
        if use_dropout:
            seq.add(K.layers.Dropout(rate))

        seq.add(K.layers.ReLU())
        return seq
        

class Discriminator(K.Model):

    def __init__(self):

        kwargs = dict(kernel_size=(4, 4),
                      strides=2,
                      padding='same')
        filter_sizes = [64, 128, 256, 512]
        self.encoders = [K.layers.Conv2D(
            filter_sizes=filter_size, name=f'conv_{i + 1}', **kwargs)\
                    for i, filter_size in enumerate(filter_sizes)]
        self.concate = K.layers.Concatenate()
        
    def call(self, inputs, targets):
        x = self.concate([inputs, targets])
        for encoder in self.encoders:
            x = encoder(x)
        return x
        
    
if __name__ == '__main__':

    inputs = K.layers.Input(shape=[256, 256, 3])
    g = Generator()
    g(inputs)

    from IPython import embed
    embed()
