#!/usr/bin/env python
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from glob import glob
import sys
import matplotlib.image as mpimg


class kaggle_cifar10(DenseDesignMatrix):

    def __init__(self, s, one_hot=False, start_idx=0, max_count=None,
                 datapath=None, axes=('c', 0, 1, 'b')):
        self.img_shape = (3, 32, 32)
        self.start_idx = start_idx
        self.max_count = max_count if max_count is not None else sys.maxsize
        self.img_size = np.prod(self.img_shape)
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.n_classes = len(self.label_names)
        self.label_map = {k: v for k, v in zip(self.label_names,
                                               range(self.n_classes))}
        self.label_unmap = {v: k for k, v in zip(self.label_names,
                                                 range(self.n_classes))}
        self.one_hot = one_hot

        if datapath is not None:
            if datapath[-1] != '/':
                datapath += '/'

        if s == 'train':
            print "Loading training set"
            files = glob(datapath + 'train/*.png')
        elif s == 'test':
            files = glob(datapath + 'test/*.png')
        else:
            raise ValueError("Only train and test data is available")
        assert len(files) > 0, "Unable to read files! Ensure that datapath \
                points to a directory containing 'train' and 'test' dirs."
        files = files[self.start_idx:self.start_idx + self.max_count]
        "Total number of files:", len(files)
        "Starting from file:", files[0], "with index", self.start_idx

        # Sort the files so they match the labels
        files = sorted(files, key=lambda x: int(x.split("/")[-1][:-4]))
        X = np.array([mpimg.imread(f) for f in files])
        X *= 255.0
        X = X.swapaxes(0, 3)

        def get_labels():
            y = np.genfromtxt(
                datapath + 'trainLabels.csv',
                delimiter=',',
                skip_header=1,
                converters={1: self.convert})
            # y is a nparray of tuples? may need fixing for non one_hot
            # scenario
            if self.one_hot:
                hot = np.zeros((y.shape[0], self.n_classes), dtype='float32')
                for i in xrange(y.shape[0]):
                    hot[i, y[i][1]] = 1.
                y = hot
            return y

        if s == 'train':
            y = get_labels()
        else:
            print "Warning: no labels for any dataset besides train!"
            y = np.zeros(X.shape)

        super(kaggle_cifar10, self).__init__(y=y,
                                             topo_view=X,
                                             axes=axes)

    def convert(self, x):
        return self.label_map[x]

    def unconvert(self, x):
        return self.label_unmap[x]
