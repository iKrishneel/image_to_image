#!/usr/bin/env python

import argparse
from  datetime import datetime
import os
import os.path as osp
import tensorflow as tf
from tqdm import tqdm, trange

from image_to_image import PatchGAN, Dataloader


def get_name():
    return datetime.now().strftime("%Y%m%d%H%M%S")

class Trainer(PatchGAN):

    def __init__(self, dataset_dir, log_path, epochs=100):
        super(Trainer, self).__init__()

        # dataset
        train_dir = osp.join(dataset_dir, 'train')
        val_dir = osp.join(dataset_dir, 'val')
        input_shape = (256, 256, 3)
        self.train_dl = Dataloader(train_dir, input_shape=input_shape)
        self.val_dl = Dataloader(val_dir, input_shape=input_shape)

        print([self.train_dl.size, self.val_dl.size])
        
        # logging directory
        assert osp.isdir(log_path), 'Invalid log directory'
        self.log_dir = osp.join(log_path, get_name())
        os.mkdir(self.log_dir)
        
        self._epochs = epochs

        self._summary = tf.summary.create_file_writer(self.log_dir)
        self._checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)
        
    def fit(self):
        checkpoint_prefix = osp.join(self.log_dir, 'checkpoint')
        iter_per_epoch = 500
        for epoch in trange(self._epochs, desc='PatchGAN'):
            for i in trange(iter_per_epoch, desc='iteration'):
                im_rgb, im_lab = self.train_dl()
                
                im_rgb = tf.expand_dims(im_rgb, axis=0)
                im_lab = tf.expand_dims(im_lab, axis=0)
                gen_losses, disc_loss = self.train_step(
                    im_lab, im_rgb, epoch=epoch)

                with self._summary.as_default():
                    tf.summary.scalar(
                        'gen_total_loss', gen_losses[0], step=epoch)
                    tf.summary.scalar(
                        'gen_gan_loss', gen_losses[1], step=epoch)
                    tf.summary.scalar(
                        'gen_l1_loss', gen_losses[2], step=epoch)
                    tf.summary.scalar(
                        'disc_loss', disc_loss, step=epoch)

            if (epoch + 1) % 20 == 0:
                self._checkpoint.save(file_prefix=checkpoint_prefix)            
        self._checkpoint.save(file_prefix=checkpoint_prefix)
        
                
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument(
        '--log_dir', type=str, required=False, default='../logs/')
    parser.add_argument('--epochs', type=int, required=False, default=100)
    args = parser.parse_args()
    
    trainer = Trainer(args.dataset, args.log_dir, args.epochs)
    trainer.fit()
