import cPickle, gzip, numpy
import theano
import theano.tensor as T

class SumRegression(object):
    def __init__(self,input,n_in,n_out,rng):
        self.W = theano.shared(value=numpy.ones((n_in,n_out),dtype=theano.config.floatX)
                               ,borrow=True,name='W')
        self.predict_prob = T.nnet.softmax(T.dot(input,self.W))
        self.predict_y = T.argmax(self.predict_prob,axis=1)
        self.params=[]

    def negative_log_likelihood(self, y):
        #return - T.mean(y * T.log(self.predict_prob) + (1 - y) * T.log(1 - self.predict_prob))
        return -T.mean(T.log(self.predict_prob)[T.arange(y.shape[0]), y])

    def errors(self,y):
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.predict_y,y))
        else:
            raise NotImplementedError

class LogisticRegression(object):
    def __init__(self,input,n_in,n_out,rng):
        self.W = theano.shared( numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ))
        self.b = theano.shared(value=numpy.zeros(n_out,dtype=theano.config.floatX),borrow=True,name='b')
        self.predict_prob = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.predict_y = T.argmax(self.predict_prob,axis=1)
        self.params=[self.W,self.b]

    def negative_log_likelihood(self, y):
        #return - T.mean(y * T.log(self.predict_prob) + (1 - y) * T.log(1 - self.predict_prob))
        return -T.mean(T.log(self.predict_prob)[T.arange(y.shape[0]), y])

    def errors(self,y):
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.predict_y,y))
        else:
            raise NotImplementedError

def load_data(dataset):
    def shared_data(data_xy):
        data_x,data_y = data_xy
        shared_x = theano.shared(data_x)
        shared_y = theano.shared(data_y)
        return shared_x, T.cast(shared_y,'int32')

    f = gzip.open(dataset)
    train_set, dev_set, test_set = cPickle.load(f)
    f.close()

    train_set_x, train_set_y = shared_data(train_set)
    dev_set_x, dev_set_y = shared_data(dev_set)
    test_set_x, test_set_y = shared_data(test_set)

    rval = [(train_set_x,train_set_y),(dev_set_x, dev_set_y ),
    (test_set_x,test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    data = load_data('mnist.pkl.gz')
    train_x, train_y = data[0]
    dev_x, dev_y = data[1]
    test_x, test_y = data[2]

    n_train_batches = train_x.get_value(borrow=True).shape[0]//batch_size
    n_dev_batches = dev_x.get_value(borrow=True).shape[0]//batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0]//batch_size

    print n_dev_batches

    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x,n_in=28*28,n_out=10)
    cost = classifier.negative_log_likelihood(y)
    print 'building model...'
    index = T.lscalar()

    g_w = T.grad(cost=cost, wrt= classifier.W)
    g_b = T.grad(cost = cost,wrt = classifier.b)
    updates = [(classifier.W,classifier.W-learning_rate*g_w),
               (classifier.b,classifier.b-learning_rate*g_b)]
    train_model = theano.function(inputs=[index],outputs = cost,updates=updates,
                                  givens={
                                      x: train_x[index*batch_size:(index+1)*batch_size],
                                      y: train_y[index*batch_size:(index+1)*batch_size]
                                  })

    validate_model = theano.function(inputs=[index],outputs = classifier.error(y),
                                  givens={
                                      x: test_x[index*batch_size:(index+1)*batch_size],
                                      y: test_y[index*batch_size:(index+1)*batch_size]
                                  })
    epoch = 0
    while epoch < n_epochs:
        epoch = epoch +1
        train_error = 0
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            #print minibatch_avg_cost
            train_error = train_error+ minibatch_avg_cost
            if minibatch_index == n_train_batches-1:
                validation_losses = [validate_model(i) for i in range(n_dev_batches)]
                this_validation_losses = numpy.mean(validation_losses)
                #print validation_losses
                print('epoch %i, minibatch %i, valiadation error %f'%(epoch,minibatch_index+1,this_validation_losses))

if __name__ == '__main__':
    sgd_optimization_mnist()