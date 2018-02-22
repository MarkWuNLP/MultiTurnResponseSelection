import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import sys, re, cPickle, random, logging, argparse
import datetime

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def kmaxpooling(input,input_shape,k):
    sorted_values = T.argsort(input,axis=3)
    topmax_indexes = sorted_values[:,:,:,-k:]
    # sort indexes so that we keep the correct order within the sentence
    topmax_indexes_sorted = T.sort(topmax_indexes)

    #given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
    dim0 = T.arange(0,input_shape[0]).repeat(input_shape[1]*input_shape[2]*k)
    dim1 = T.arange(0,input_shape[1]).repeat(k*input_shape[2]).reshape((1,-1)).repeat(input_shape[0],axis=0).flatten()
    dim2 = T.arange(0,input_shape[2]).repeat(k).reshape((1,-1)).repeat(input_shape[0]*input_shape[1],axis=0).flatten()
    dim3 = topmax_indexes_sorted.flatten()
    return input[dim0,dim1,dim2,dim3].reshape((input_shape[0], input_shape[1], input_shape[2], k))

class QALeNetConvPoolLayer(object):
    """ Convolution Layer and Pool Layer for Question and Sentence pair """

    def __init__(self, rng, linp, rinp, filter_shape, poolsize):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type linp: theano.tensor.TensorType
        :param linp: symbolic variable that describes the left input of the
        architecture (one minibatch)

        :type rinp: theano.tensor.TensorType
        :param rinp: symbolic variable that describes the right input of the
        architecture (one minibatch)

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, 1,
                              filter height,filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        self.linp = linp
        self.rinp = rinp
        self.filter_shape = filter_shape
        self.poolsize = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")

        # convolve input feature maps with filters
        lconv_out = conv.conv2d(input=linp, filters=self.W, filter_shape = filter_shape)
        rconv_out = conv.conv2d(input=rinp, filters=self.W, filter_shape = filter_shape)
        self.lconv_out_tanh = ReLU(lconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.rconv_out_tanh = ReLU(rconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.loutput = theano.tensor.signal.pool.pool_2d(input=self.lconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="max")
        self.routput = theano.tensor.signal.pool.pool_2d(input=self.rconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="max")
        self.params = [self.W, self.b]

    def predict(self, lnew_data, rnew_data):
        """
        predict for new data
        """
        lconv_out = conv.conv2d(input=lnew_data, filters=self.W)
        rconv_out = conv.conv2d(input=rnew_data, filters=self.W)
        lconv_out_tanh = T.tanh(lconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        rconv_out_tanh = T.tanh(rconv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        loutput = theano.tensor.signal.pool.pool_2d(input=lconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="max")
        routput = theano.tensor.signal.pool.pool_2d(input=rconv_out_tanh, ds=self.poolsize, ignore_border=True, mode="max")
        return loutput, routput

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        print 'image shape', image_shape
        print 'filter shape', filter_shape
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape),
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),borrow=True,name="W_conv")
        b_values =np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True,mode="max")
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]

    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output =theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output

class LeNetConvPoolLayer2(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        print 'image shape', image_shape
        print 'filter shape', filter_shape
        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape),
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),borrow=True,name="W_conv")
        b_values =np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        self.params = [self.W, self.b]
        # convolve input feature maps with filters


    def __call__(self, input):
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output =theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True,mode="max")
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output


    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output