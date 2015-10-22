# -*- coding: utf-8 -*-
"""
Don't install lasagne, theano and nolearn using pip, use the following:
    
pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git

http://martin-thoma.com/lasagne-for-python-newbies/#tocAnchor-1-4
"""

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print 'Looping %d times took' % iters, t1 - t0, 'seconds'
print 'Result is', r
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')

