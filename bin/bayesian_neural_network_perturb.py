import edward as ed
from edward.models import Normal
from math import ceil
import numpy as np
import tensorflow as tf

from utils import tprint

def neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2):
    h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
    h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return h#tf.reshape(h, [-1])

class BayesianNN(object):
    def __init__(
            self,
            n_hidden1=200,
            n_hidden2=200,
            n_iter=1000,
            n_posterior_samples=100,
            batch_size=500,
            sketch_size=None,
            random_state=None,
            verbose=False,
    ):
        if random_state is not None:
            ed.set_seed(random_state)
            np.random.seed(random_state)
            tf.set_random_seed(random_state)

        self.n_hidden1_ = n_hidden1
        self.n_hidden2_ = n_hidden2
        self.n_iter_ = n_iter
        self.n_posterior_samples_ = n_posterior_samples
        self.batch_size_ = batch_size
        self.sketch_size_ = sketch_size
        self.verbose_ = verbose

    def fit(self, X, y):
        if self.sketch_size_ is not None:
            from fbpca import pca
            from geosketch import gs

            if X.shape[1] > 100:
                U, s, _ = pca(X, k=100)
                X_dimred = U * s
            else:
                X_dimred = X
            gs_idx = gs(X_dimred, self.sketch_size_, replace=False)
            X = X[gs_idx]
            y = y[gs_idx]

        n_samples, n_features = X.shape

        if self.verbose_:
            tprint('Fitting Bayesian NN on {} data points with dimension {}...'
                   .format(*X.shape))

        X = X.astype(np.float32) # Edward uses float32.

        # Bayesian weights.

        W_0_shape = [ n_features, self.n_hidden1_ ]
        W_0 = Normal(loc=tf.zeros(W_0_shape), scale=tf.ones(W_0_shape))

        W_1_shape = [ self.n_hidden1_, self.n_hidden2_ ]
        W_1 = Normal(loc=tf.zeros(W_1_shape), scale=tf.ones(W_1_shape))

        W_2_shape = [ self.n_hidden2_, y.shape[1] ]
        W_2 = Normal(loc=tf.zeros(W_2_shape), scale=tf.ones(W_2_shape))

        # Bayesian biases.

        b_0 = Normal(loc=tf.zeros(self.n_hidden1_),
                     scale=tf.ones(self.n_hidden1_))
        b_1 = Normal(loc=tf.zeros(self.n_hidden2_),
                     scale=tf.ones(self.n_hidden2_))
        b_2 = Normal(loc=tf.zeros(y.shape[1]), scale=tf.ones(y.shape[1]))

        # Approximating distributions for KL divergence
        # variational inference.

        qW_0 = Normal(
            loc=tf.get_variable("qW_0/loc", W_0_shape),
            scale=tf.nn.softplus(tf.get_variable("qW_0/scale", W_0_shape))
        )
        qW_1 = Normal(
            loc=tf.get_variable("qW_1/loc", W_1_shape),
            scale=tf.nn.softplus(tf.get_variable("qW_1/scale", W_1_shape))
        )
        qW_2 = Normal(
            loc=tf.get_variable("qW_2/loc", W_2_shape),
            scale=tf.nn.softplus(tf.get_variable("qW_2/scale", W_2_shape))
        )
        qb_0 = Normal(
            loc=tf.get_variable("qb_0/loc", [self.n_hidden1_]),
            scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [self.n_hidden1_]))
        )
        qb_1 = Normal(
            loc=tf.get_variable("qb_1/loc", [self.n_hidden2_]),
            scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [self.n_hidden2_]))
        )
        qb_2 = Normal(
            loc=tf.get_variable("qb_2/loc", [y.shape[1]]),
            scale=tf.nn.softplus(tf.get_variable("qb_2/scale", [y.shape[1]]))
        )

        # Fit model.

        X_variational = tf.placeholder(
            tf.float32, [ n_samples, n_features ], name='X'
        )

        y_variational = Normal(
            loc=neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2),
            scale=tf.ones((n_samples, y.shape[1]))
        )

        inference = ed.KLqp(
            { W_0: qW_0, b_0: qb_0,
              W_1: qW_1, b_1: qb_1,
              W_2: qW_2, b_2: qb_2, },
            data={ X_variational: X, y_variational: y }
        )

        self.sess_ = ed.get_session()
        tf.global_variables_initializer().run()

        inference.run(n_iter=self.n_iter_, n_samples=10)

        self.model_ = {
            'qW_0': qW_0, 'qb_0': qb_0,
            'qW_1': qW_1, 'qb_1': qb_1,
            'qW_2': qW_2, 'qb_2': qb_2,
        }

        if self.verbose_:
            tprint('Done fitting Bayesian NN model.')

        return self

    def predict(self, X):
        if self.verbose_:
            tprint('Finding Bayesian NN predictions on {} data points...'
                   .format(X.shape[0]))

        X = X.astype(np.float32) # Edward uses float32.

        qW_0 = self.model_['qW_0']
        qW_1 = self.model_['qW_1']
        qW_2 = self.model_['qW_2']

        qb_0 = self.model_['qb_0']
        qb_1 = self.model_['qb_1']
        qb_2 = self.model_['qb_2']

        samp = [
            (qW_0.sample(), qW_1.sample(), qW_2.sample(),
             qb_0.sample(), qb_1.sample(), qb_2.sample())
            for _ in range(self.n_posterior_samples_)
        ]

        pred = []
        n_batches = int(ceil(float(X.shape[0]) / self.batch_size_))

        for batch_num in range(n_batches):
            start = batch_num*self.batch_size_
            end = (batch_num+1)*self.batch_size_
            X_batch = X[start:end]

            samples = tf.stack([
                neural_network(
                    X_batch, samp[s][0], samp[s][1], samp[s][2],
                    samp[s][3], samp[s][4], samp[s][5],
                )
                for s in range(self.n_posterior_samples_)
            ])
            pred.append(samples.eval())

            if self.verbose_:
                tprint('Finished predicting batch number {}/{}'
                       .format(batch_num + 1, n_batches))

        pred = np.hstack(pred)

        if self.verbose_:
            tprint('Done predicting with Bayesian NN model.')

        assert(pred.shape[0] == self.n_posterior_samples_)
        assert(pred.shape[1] == X.shape[0])

        self.uncertainties_ = pred.var(0).mean(1)#.flatten()
        return pred.mean(0)#.flatten()
