#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import pkg_resources
import sys

from setuptools import find_packages
from setuptools import setup


setup_requires = []
install_requires = [
    'opencv-python>=4.1.2.30',
    'numpy>=1.16',
    'matplotlib>=2.1.2',
    'keras',
    'tensorflow',
    'tqdm',
]


setup(
    name='image_to_image',
    version='0.0.0',
    description='',
    author='Krishneel Chaudhary',
    author_email='krishneel@krishneel',
    license='None',
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=install_requires
)
