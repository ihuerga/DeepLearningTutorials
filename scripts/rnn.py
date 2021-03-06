import preprocessData as p
import theano, numpy as np
from theano import tensor as T
from theano import pp
import logging
import random

logger = logging.getLogger(__name__)


class RNN(object):

    def __init__(self, n_inputs, n_hidden, n_output,
                 activation=T.tanh, L1_reg=0.0001, L2_reg=0.0001,
                 window=2):

        '''

        :param n_inputs: number of input units
        :param n_hidden: number of hidden units
        :param n_output: number of output units
        :return:
        '''

        # parameters
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg
        self.wSize = window

        # theta1, weight matrix from input to hidden units
        self.theta1 = theano.shared(name='theta1',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (n_inputs, n_hidden))
                                    .astype(theano.config.floatX))

        # thetah, recurrent weights matrix (hidden to hidden)
        self.thetah1 = theano.shared(name='thetah1',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (n_hidden, n_hidden))
                                    .astype(theano.config.floatX))

        self.thetah2 = theano.shared(name='thetah2',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (n_hidden, n_hidden))
                                    .astype(theano.config.floatX))

        # theta2, weight matrix from hiddent to output units
        self.theta2 = theano.shared(name='theta2',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (n_hidden, n_output))
                                    .astype(theano.config.floatX))

        # bh, bias vector for hidden units
        self.bh = theano.shared(name='bh',
                                value=np.zeros(n_hidden,
                                dtype=theano.config.floatX))

        # bout, bias vector for output units
        self.bout = theano.shared(name='bout',
                                  value=np.zeros(n_output,
                                  dtype=theano.config.floatX))

        # h0, hidden states
        self.h0 = theano.shared(name='h0',
                                value=np.zeros((2,n_hidden),
                                dtype=theano.config.floatX))

        # all the parameters
        self.params = [self.theta1, self.thetah1, self.thetah2,
                       self.theta2, self.bh, self.bout, self.h0]

        # activation function
        self.activation = activation

        # symbolic links to the inputs (for a minibatch)
        x = T.matrix('x', dtype=theano.config.floatX)
        y = T.matrix('y', dtype=theano.config.floatX)

        # forward pass with recurrence y_t_minus_1 (only 1 time step in context window)
        def forward_pass(x_t, h_tm2, h_tm1):

            h_t = self.activation(T.dot(x_t, self.theta1) +
                                  (T.dot(h_tm2, self.thetah1) + self.bh) +
                                  (T.dot(h_tm1, self.thetah2) + self.bh)
                                  )

            # output at t
            y_t = T.dot(h_t, self.theta2) + self.bout

            return h_t, y_t

        [h, y_pred], _ = theano.scan(fn=forward_pass,
                                sequences=x,
                                outputs_info=[dict(initial=self.h0, taps=[-2,-1]), None],
                                n_steps=x.shape[0])

        # let's use L1 and L2 regularization

        # L1 regularization
        self.L1 = 0
        self.L1 += abs(self.thetah1.sum())
        self.L1 += abs(self.thetah2.sum())
        self.L1 += abs(self.theta1.sum())
        self.L1 += abs(self.theta2.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.thetah1 ** 2).sum()
        self.L2_sqr += (self.thetah2 ** 2).sum()
        self.L2_sqr += (self.theta1 ** 2).sum()
        self.L2_sqr += (self.theta2 ** 2).sum()


        # L2 regularization

        # learning rate
        lr = T.scalar('lr', dtype=theano.config.floatX)
        self.y_prediction = y_pred
        # cost function for a real number prediction
        cost = T.mean((y_pred - y)) ** 2 + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        cost_printed = theano.printing.Print('cost')(cost)

        # gradients
        grad = [T.grad(cost, param) for param in self.params]

        # update the parameters
        updates = [ (p, p - lr * g) for p, g in zip(self.params, grad)]
        #updates_printed = theano.printing.Print('updates')(updates)

        self.batch_train = theano.function(inputs=[x,y,lr], outputs=cost, updates=updates)
        self.batch_train_print = theano.function(inputs=[x,y,lr], outputs=cost_printed, on_unused_input='warn')

        self.loss = lambda y: self.mse(y)

        self.compute_training_error = theano.function(inputs=[x,y], outputs=self.loss(y))


    def train(self, x, y, learning_rate):
        self.batch_train(x, y, learning_rate)
        self.batch_train_print(x, y, learning_rate)

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_prediction - y) ** 2)



def main(params=None):
    if not params:
        params = {
            'lr' : 0.01,
            'decay' : True,
            'n_hidden' : 50,
            'seed' : 234,
            'epochs' : 2
        }

    print params

    training_data, validation_data, testing_data = p.getdata()
    training_data_x, training_data_y  = training_data
    testing_data_x, testing_data_y = testing_data
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    rnn = RNN(n_inputs=13, n_hidden=params['n_hidden'], n_output=1)

    epoch = 0
    n_epoch = params['epochs']
    while (epoch < n_epoch):
        epoch += 1
        total_cost = 0;
        for batch_index in xrange(training_data_x.__len__()):
            rnn.train(training_data_x[batch_index], training_data_y[batch_index], params['lr'])
        train_loss = [rnn.compute_training_error(testing_data_x[i], testing_data_y[i]) for i in xrange(testing_data_x.__len__())]
        total_training_loss = np.mean(train_loss)
        print 'epoch %i with test mse %f' % (epoch, total_training_loss)

if __name__ == '__main__':
    main()




