
import os
import sys
import timeit

import numpy
from CNN import QALeNetConvPoolLayer,LeNetConvPoolLayer2
from Classifier import HiddenLayer2
import theano
import theano.tensor as T
import numpy as np
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')
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

class PoolingSim(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.W = theano.shared(value=ortho_weight(100), name='W', borrow=True)
        self.activation = activation
        self.hidden_layer = HiddenLayer2(rng,2*5*n_in,n_out)

        self.params = [self.W] + self.hidden_layer.params

    def __call__(self, input_l,input_r,batch_size,max_l):
        channel_1 = T.batched_dot(input_l,input_r.dimshuffle(0,2,1))
        channel_2 = T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))
        input = T.stack([channel_1,channel_2],axis=1)
        poolingoutput = kmaxpooling(input,[batch_size,2,max_l,max_l],5)
        mlp_in = T.flatten(poolingoutput,2)
        return self.hidden_layer(mlp_in)

class PoolingSim3(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None,
                 activation=T.tanh,hidden_size=100):
        self.W = theano.shared(value=ortho_weight(hidden_size), name='W', borrow=True)
        self.activation = activation
        self.hidden_layer = HiddenLayer2(rng,2*5*n_in,n_out)

        self.params = [self.W] + self.hidden_layer.params

    def __call__(self,origin_l,origin_r,input_l,input_r,batch_size,max_l):
        channel_1 = T.batched_dot(origin_l,origin_r.dimshuffle(0,2,1))
        channel_2 = T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))
        input = T.stack([channel_1,channel_2],axis=1)
        poolingoutput = kmaxpooling(input,[batch_size,2,max_l,max_l],5)
        mlp_in = T.flatten(poolingoutput,2)
        return self.hidden_layer(mlp_in)

class PoolingSim2(object):
    def __init__(self, rng,  n_in, n_out,tensor_num = 3,
                 activation=T.tanh):
        self.tensor_num = tensor_num
        self.W = []
        for i in range(tensor_num):
            self.W.append(theano.shared(value=ortho_weight(100), borrow=True))
        self.activation = activation
        self.hidden_layer = HiddenLayer2(rng,tensor_num*5*n_in,n_out)

        self.params = self.W + self.hidden_layer.params

    def __call__(self, input_l,input_r,batch_size,max_l):
        channels = []
        for i in range(self.tensor_num):
            channels.append(T.batched_dot(T.dot(input_l,self.W[i]),input_r.dimshuffle(0,2,1)))

        input = T.stack(channels,axis=1)
        poolingoutput = kmaxpooling(input,[batch_size,self.tensor_num,max_l,max_l],5)
        mlp_in = T.flatten(poolingoutput,2)
        return self.hidden_layer(mlp_in)

class ConvSim(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None,
                 activation=T.tanh,hidden_size=100):
        self.W = theano.shared(value=ortho_weight(hidden_size), borrow=True)
        self.activation = activation

        self.conv_layer = LeNetConvPoolLayer2(rng,filter_shape=(8,2,3,3),
                                    image_shape=(200,2,50,50)
                       ,poolsize=(3,3),non_linear='relu')

        self.hidden_layer = HiddenLayer2(rng,2048,n_out)
        self.params = [self.W,] + self.conv_layer.params + self.hidden_layer.params
    def Get_M2(self,input_l,input_r):
        return T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))

    def __call__(self, origin_l,origin_r,input_l,input_r):
        channel_1 = T.batched_dot(origin_l,origin_r.dimshuffle(0,2,1))
        channel_2 = T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))
        input = T.stack([channel_1,channel_2],axis=1)
        mlp_in = T.flatten(self.conv_layer(input),2)

        return self.hidden_layer(mlp_in)

class ConvSim2(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None,
                 activation=T.tanh,hidden_size=100):
        self.W = theano.shared(value=ortho_weight(hidden_size), borrow=True)
        self.activation = activation

        self.conv_layer = LeNetConvPoolLayer2(rng,filter_shape=(8,1,3,3),
                                    image_shape=(200,1,50,50)
                       ,poolsize=(3,3),non_linear='relu')

        self.hidden_layer = HiddenLayer2(rng,2048,n_out)
        self.params = self.conv_layer.params + self.hidden_layer.params

    def __call__(self, origin_l,origin_r):
        channel_1 = T.batched_dot(origin_l,origin_r.dimshuffle(0,2,1))
        input =channel_1.dimshuffle(0,'x',1,2)
        mlp_in = T.flatten(self.conv_layer(input),2)

        return self.hidden_layer(mlp_in)