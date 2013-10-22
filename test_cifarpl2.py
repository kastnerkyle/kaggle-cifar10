#!/usr/bin/env python
from pylearn2.utils import serial
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing

tst = cifar10.CIFAR10('test',
                      toronto_prepro=False,
                      one_hot=True,
                      axes=('c', 0, 1, 'b'))

# preprocessor = preprocessing.ZCA()
# tst.apply_preprocessor(preprocessor=preprocessor, can_fit=True)

mdl = serial.load('cifar10_maxout_zca.pkl')
batch_size = 20
mdl.set_batch_size(batch_size)
X = mdl.get_input_space().make_batch_theano()
Y = mdl.fprop(X)

from theano import tensor as T
from theano import function
y = T.argmax(Y, axis=1)
f = function([X], y)
y = []
for i in xrange(tst.X.shape[0] / batch_size):
    x_arg = tst.X[i * batch_size:(i + 1) * batch_size, :]
    if X.ndim > 2:
        x_arg = tst.get_topological_view(x_arg)
    y.append(f(x_arg))
print y
