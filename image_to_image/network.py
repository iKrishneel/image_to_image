#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras as K

def conv(batch_norm=True, **kwargs):
    slope = kwargs.get('slope', 0.2)
    seq = K.Sequential()
    seq.add(K.layers.Conv2D(**kwargs))
        
    if batch_norm:
        seq.add(K.layers.BatchNormalization())
        seq.add(K.layers.LeakyReLU(slope))
    return seq


def deconv(batch_norm=True, use_dropout=True, **kwargs):
    rate = kwargs.pop('dropout_rate', 0.5)
    seq = K.Sequential()
    seq.add(K.layers.Conv2DTranspose(**kwargs))
        
    if batch_norm:
        seq.add(K.layers.BatchNormalization())
            
    if use_dropout:
        seq.add(K.layers.Dropout(rate))

    seq.add(K.layers.ReLU())
    return seq

    
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
          [conv(batch_norm=bn, filters=fsize,
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
          [deconv(use_dropout=do, filters=fsize, kernel_size=ksize,
                  strides=s, padding='same', use_bias=False,
                  kernel_initializer=initializer,
                  name=f'deconv{i+1}')
           for i, (fsize, ksize, s, do) in enumerate(dfilter_sizes)]

        self.concate = K.layers.Concatenate()
        self.top = K.layers.Conv2DTranspose(
            output_channels, 4, strides=2, padding='same',
            kernel_initializer=initializer, activation='tanh')

    def call(self, inputs):
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
        

class Discriminator(K.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        filter_sizes = [[64, [4, 4], 2, False],
                        [128, [4, 4], 2, True],
                        [256, [4, 4], 2, True],
                        [512, [4, 4], 2, True],
                        [512, [4, 4], 1, True]]
        initializer = tf.random_normal_initializer(0., 0.02)
        self.encoders = \
          [conv(batch_norm=bn, filters=fsize, kernel_size=(4, 4),
                kernel_initializer=initializer,
                strides=s,padding='same', name=f'conv{i+1}')
           for i, (fsize, ksize, s, bn) in enumerate(filter_sizes)]
        
        self.concate = K.layers.Concatenate()

        self.classifier = K.layers.Conv2D(
            filters=1, kernel_size=(4, 4), padding='same')
        
    def call(self, inputs, targets):
        x = self.concate([inputs, targets])
        for encoder in self.encoders:
            x = encoder(x)
        x = K.activations.sigmoid(self.classifier(x))
        return x

    
class PatchGAN(object):

    def __init__(self):

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.gen_optimizer = K.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_optimizer = K.optimizers.Adam(2e-4, beta_1=0.5)
        
        self._entropy = K.losses.BinaryCrossentropy(
            from_logits=True)
        self._lambda = 100

    def discriminator_loss(self, real, generated):
        real_loss = self._entropy(tf.ones_like(real), real)
        gen_loss = self._entropy(tf.zeros_like(generated), generated)
        return real_loss + gen_loss

    def generator_loss(self, generated, gen_out, target):
        gan_loss = self._entropy(tf.ones_like(generated), generated)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_out))
        total_gen_loss = gan_loss + (lambda_val * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape,\
            tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            dreal_output = self.discriminator(
                input_image, target, training=True)
            dgen_output = self.discriminator(
                input_image, gen_output, training=True)

            gen_losses = self.generator_loss(
                dgen_output, gen_output, target)
            # gen_total_loss, gen_gan_loss, gen_l1_loss = gen_losses
            disc_loss = self.discriminator_loss(dreal_output, dgen_output)

            generator_gradients = gen_tape.gradient(
                gen_losses[0], self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            self.gen_optimizer.apply_gradients(
                zip(generator_gradients,
                    self.generator.trainable_variables))
            self.disc_optimizer.apply_gradients(
                zip(discriminator_gradients,
                    self.discriminator.trainable_variables))

            return gen_losses, disc_loss

    
if __name__ == '__main__':

    inputs = K.layers.Input(shape=[256, 256, 3])
    g = Generator()
    g(inputs)

    d = Discriminator()
    # d(inputs, inputs)

    from IPython import embed
    embed()
