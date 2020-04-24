#!/usr/bin/env python

import os
import os.path as osp
import numpy as np
from random import randint
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt


def is_valid_dir(directory):
    assert osp.isdir(directory),\
      f'Invalid dataset directory {directory}'


def read_dir(directory):
    return [osp.join(directory, f) for f in os.listdir(directory)
            if f.split('.')[1] in ['jpg', 'png']]


class Dataloader(object):

    def __init__(self, directory: str, input_shape=(256, 256, 3)):
        is_valid_dir(directory)
        self._dataset = read_dir(directory)
        self._input_shape = input_shape

    def __call__(self, index=None):
        index = randint(0, len(self._dataset)) \
          if index is None else index
        im_rgb, im_lab = self.randomize(*self.load(index=index))
        im_rgb = self.normalize(im_rgb)
        im_lab = self.normalize(im_lab)
        return im_rgb, im_lab
        
    def load(self, index):
        im = cv.imread(self._dataset[index], cv.IMREAD_COLOR)
        shape = int(im.shape[1] // 2)
        im_rgb = im[:, :shape].astype(np.float32)
        im_lab = im[:, shape:].astype(np.float32)
        return im_rgb, im_lab

    def normalize(self, im: np.ndarray) -> np.ndarray:
        # im = im.astype(np.float32)
        m, n = np.min(im), np.max(im)
        return (im - m) / (n - m)

    def resize(self, im: np.ndarray, shape=None):
        shape = self._input_shape[:2] if shape is None else shape
        return tf.image.resize(
            im, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def random_crop(self, im_rgb: np.ndarray, im_lab: np.ndarray):
        return tf.image.random_crop(
            tf.stack([im_rgb, im_lab], axis=0), size=[2, *self._input_shape])

    def randomize(self, im_rgb: np.ndarray, im_lab: np.ndarray):
        new_shape = (286, 286)
        im_rgb = self.resize(im_rgb, new_shape)
        im_lab = self.resize(im_lab, new_shape)
        im_rgb, im_lab = self.random_crop(im_rgb, im_lab)
        if tf.random.uniform(()) > 0.5:
            im_rgb = tf.image.flip_left_right(im_rgb)
            im_lab = tf.image.flip_left_right(im_lab)
        return im_rgb, im_lab
    
    @property
    def dataset(self):
        return self._dataset

    @property
    def size(self):
        return len(self._dataset)

if __name__ == '__main__':
    d = Dataloader('/Users/krishneelchaudhary/Downloads/facades/train/')
    d(10)
