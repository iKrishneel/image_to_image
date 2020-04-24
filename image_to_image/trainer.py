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
        train_dl = Dataloader(train_dir, input_shape=input_shape)
        val_dl = Dataloader(val_dir, input_shape=input_shape)

        print([train_dl.size, val_dl.size])
        
        # logging directory
        assert osp.isdir(log_path), 'Invalid log directory'
        log_dir = osp.join(log_path, get_name())
        os.mkdir(log_dir)
        print(f'Logging path {log_dir}')
        
        self._epochs = epochs
        
    def fit(self):

        for epoch in trange(self._epochs, desc='PatchGAN'):
            input_image = 1
            target = 1
            losses = self.train_step(input_image, target, epoch=epoch)
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument(
        '--log_dir', type=str, required=False, default='../logs/')
    parser.add_argument('--epochs', type=int, required=False, default=100)
    args = parser.parse_args()
    
    trainer = Trainer(args.dataset, args.log_dir, args.epochs)
    # trainer.fit()
