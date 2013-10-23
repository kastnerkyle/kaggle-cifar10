#!/usr/bin/env python
from kaggle_dataset import kaggle_cifar10
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets import preprocessing
import numpy as np

t1 = kaggle_cifar10('train',
                    one_hot=True)

print t1.X.shape
# preprocessor = preprocessing.ZCA()
# t1.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
# t2.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
