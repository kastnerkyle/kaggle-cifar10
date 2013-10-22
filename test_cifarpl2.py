#!/usr/bin/env python
from pylearn2.utils import serial
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing
from theano import tensor as T
from theano import function
import numpy as np


trn = cifar10.CIFAR10('train',
                      toronto_prepro=False,
                      one_hot=True,
                      axes=('c', 0, 1, 'b'))

tst = cifar10.CIFAR10('test',
                      toronto_prepro=False,
                      one_hot=True,
                      axes=('c', 0, 1, 'b'))

preprocessor = preprocessing.ZCA()
trn.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
tst.apply_preprocessor(preprocessor=preprocessor, can_fit=True)

mdl = serial.load('cifar10_maxout_zca.pkl')
# This batch size must be evenly divisible into number of total samples!
batch_size = 100
mdl.set_batch_size(batch_size)
X = mdl.get_input_space().make_batch_theano()
Y = mdl.fprop(X)
y = T.argmax(Y, axis=1)
f = function([X], y)
yhat = []
for i in xrange(tst.X.shape[0] / batch_size):
    x_arg = tst.X[i * batch_size:(i + 1) * batch_size, :]
    x_arg = tst.get_topological_view(x_arg)
    yhat.append(f(x_arg))
yhat = np.array(yhat)

print "Test(%): ", (tst.y.argmax(axis=1).reshape(yhat.shape) == yhat).mean()
