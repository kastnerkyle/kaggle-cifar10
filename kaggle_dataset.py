#!/usr/bin/env python
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from glob import glob
from scipy import misc


class kaggle_cifar10(DenseDesignMatrix):

    def __init__(self, s, one_hot=False, axes=('b', 0, 1, 'c')):
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.label_map = {k: v for k, v in zip(self.label_names,
                                               range(len(self.label_names)))}
        self.label_unmap = {v: k for k, v in zip(self.label_names,
                                                 range(len(self.label_names)))}

        if s == 'train':
            files = glob(
                '/home/kkastner/kaggle_data/kaggle-cifar10/train/*.png')
        elif s == 'test':
            files = glob(
                '/home/kkastner/kaggle_data/kaggle-cifar10/test/*.png')
        else:
            raise ValueError("Only train and test datasets are available")

        X = np.array([misc.imread(f).astype('float32')
                      for f in files])
        X = X.reshape(X.shape[0], 3072)

        def convert(x):
            return self.label_map[x]
        if s == 'train':
            y = np.genfromtxt(
                '/home/kkastner/kaggle_data/kaggle-cifar10/trainLabels.csv',
                delimiter=',',
                skip_header=1,
                converters={1: convert})
            self.one_hot = one_hot
            if one_hot:
                one_hot = np.zeros((y.shape[0], 10), dtype='float32')
                for i in xrange(y.shape[0]):
                    one_hot[i, y[i][1] - 1] = 1.
                y = one_hot
        else:
            y = None
        view_converter = DefaultViewConverter((32, 32, 3), axes)
        super(kaggle_cifar10, self).__init__(X=X, y=y,
                                             view_converter=view_converter)
