#!/usr/bin/env python
from kaggle_dataset import kaggle_cifar10
from pylearn2.datasets import preprocessing

t1 = kaggle_cifar10('train',
                    one_hot=True)

preprocessor = preprocessing.ZCA()
t1.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
