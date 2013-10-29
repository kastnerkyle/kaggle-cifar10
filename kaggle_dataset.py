#!/usr/bin/env python
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from glob import glob
import matplotlib.image as mpimg


class kaggle_cifar10(DenseDesignMatrix):

    def __init__(self, s, one_hot=False, datapath=None, axes=('c', 0, 1, 'b')):
        self.img_shape = (3, 32, 32)
        self.img_size = np.prod(self.img_shape)
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        self.n_classes = len(self.label_names)
        self.label_map = {k: v for k, v in zip(self.label_names,
                                               range(self.n_classes))}
        self.label_unmap = {v: k for k, v in zip(self.label_names,
                                                 range(self.n_classes))}
        self.one_hot = one_hot

        assert datapath is not None
        # Really should check for files here...
        if datapath[-1] != '/':
            datapath += '/'

        if s == 'train':
            print "Loading training set"
            files = glob(datapath + 'train/*.png')
        elif s == 'test':
            files = glob(datapath + 'test/*.png')
        else:
            raise ValueError("Only train and test data is available")

        #Sort the files so they match the labels
        files = sorted(files, key=lambda x: int(x.split("/")[-1][:-4]))
        X = np.array([mpimg.imread(f) for f in files])
        X *= 255.0
        X = X.swapaxes(0, 3)

        def convert(x):
            return self.label_map[x]

        def get_labels():
            y = np.genfromtxt(
                datapath + 'trainLabels.csv',
                delimiter=',',
                skip_header=1,
                converters={1: convert})
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
            y = None

        super(kaggle_cifar10, self).__init__(y=y,
                                             topo_view=X,
                                             axes=axes)
