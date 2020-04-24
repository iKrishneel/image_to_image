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
        log_dir = osp.join(log_path, get_name())
        os.mkdir(log_dir)
        print(f'Logging path {log_dir}')
        
        self._epochs = epochs
        
    def fit(self):

        for epoch in trange(self._epochs, desc='PatchGAN'):
            im_rgb, im_lab = self.train_dl()

            im_rgb = tf.expand_dims(im_rgb, axis=0)
            im_lab = tf.expand_dims(im_lab, axis=0)
            losses = self.train_step(im_lab, im_rgb, epoch=epoch)
            print(losses)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument(
        '--log_dir', type=str, required=False, default='../logs/')
    parser.add_argument('--epochs', type=int, required=False, default=100)
    args = parser.parse_args()
    
    trainer = Trainer(args.dataset, args.log_dir, args.epochs)
    trainer.fit()
