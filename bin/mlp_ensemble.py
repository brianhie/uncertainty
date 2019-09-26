from math import ceil
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.activations import softplus

from utils import tprint

tf.set_random_seed(1)

def check_param_length(param, n_regressors):
    if len(param) != n_regressors:
        raise ValueError('Invalid parameter list length')

def gaussian_nll(ytrue, ypreds):
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)

    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)

class MLPEnsembleRegressor(object):
    def __init__(self,
                 layer_sizes_list,
                 activations='relu',
                 loss='mse',
                 solvers='adam',
                 alphas=0.0001,
                 batch_sizes=None,
                 max_iters=200,
                 momentums=0.9,
                 nesterovs_momentums=True,
                 backend='keras',
                 verbose=False,
    ):
        self.backend_ = backend
        self.verbose_ = verbose

        self.n_regressors_ = len(layer_sizes_list)
        self.layer_sizes_list_ = layer_sizes_list

        self.loss_ = loss

        # Activation functions.
        if issubclass(type(activations), list):
            check_param_length(activations, self.n_regressors_)
            self.activations_ = activations
        else:
            self.activations_ = [ activations ] * self.n_regressors_

        # Solvers.
        if issubclass(type(solvers), list):
            check_param_length(solvers, self.n_regressors_)
            self.solvers_ = solvers_
        else:
            self.solvers_ = [ solvers ] * self.n_regressors_

        # Alphas.
        if issubclass(type(alphas), list):
            check_param_length(alphas, self.n_regressors_)
            self.alphas_ = alphas_
        else:
            self.alphas_ = [ alphas ] * self.n_regressors_

        # Batch Sizes.
        if issubclass(type(batch_sizes), list):
            check_param_length(batch_sizes, self.n_regressors_)
            self.batch_sizes_ = batch_sizes_
        else:
            self.batch_sizes_ = [ batch_sizes ] * self.n_regressors_

        # Maximum number of iterations.
        if issubclass(type(max_iters), list):
            check_param_length(max_iters, self.n_regressors_)
            self.max_iters_ = max_iters
        else:
            self.max_iters_ = [ max_iters ] * self.n_regressors_

        # Momentums.
        if issubclass(type(momentums), list):
            check_param_length(momentums, self.n_regressors_)
            self.momentums_ = momentums_
        else:
            self.momentums_ = [ momentums ] * self.n_regressors_

        # Whether to use Nesterov's momentum.
        if issubclass(type(nesterovs_momentums), list):
            check_param_length(nesterovs_momentums, self.n_regressors_)
            self.nesterovs_momentums_ = nesterovs_momentums_
        else:
            self.nesterovs_momentums_ = [ nesterovs_momentums ] * self.n_regressors_

    def _create_models(self, X, y):
        if len(y.shape) == 1:
            n_outputs = 1
        else:
            raise ValueError('Only scalar predictions are currently supported.')

        self.models_ = []

        if self.backend_ == 'sklearn':
            from sklearn.neural_network import MLPRegressor

            for model_idx in range(self.n_regressors_):
                model = MLPRegressor(
                    hidden_layer_sizes=self.layer_sizes_list_[model_idx],
                    activation=self.activations_[model_idx],
                    solver=self.solvers_[model_idx],
                    alpha=self.alphas_[model_idx],
                    batch_size=self.batch_sizes_[model_idx],
                    max_iter=self.max_iters_[model_idx],
                    momentum=self.momentums_[model_idx],
                    nesterovs_momentum=self.nesterovs_momentums_[model_idx],
                    verbose=self.verbose_,
                )
                self.models_.append(model)

        elif self.backend_ == 'keras':
            from keras import regularizers
            from keras.layers import Dense
            from keras.models import Sequential

            for model_idx in range(self.n_regressors_):
                hidden_layer_sizes = self.layer_sizes_list_[model_idx]

                model = Sequential()
                for layer_size in hidden_layer_sizes:
                    model.add(Dense(layer_size, kernel_initializer='normal',
                                    activation=self.activations_[model_idx],
                                    kernel_regularizer=regularizers.l2(0.01)))

                if self.loss_ == 'mse':
                    model.add(
                        Dense(1, kernel_initializer='normal',
                              kernel_regularizer=regularizers.l2(0.01))
                    )
                    model.compile(loss='mean_squared_error',
                                  optimizer=self.solvers_[model_idx])

                elif self.loss_ == 'gaussian_nll':
                    model.add(
                        Dense(2, kernel_initializer='normal',
                              kernel_regularizer=regularizers.l2(0.01))
                    )
                    model.compile(loss=gaussian_nll,
                                  optimizer=self.solvers_[model_idx])

                self.models_.append(model)

    def fit(self, X, y):
        y = y.flatten()
        if len(y) != X.shape[0]:
            raise ValueError('Data has {} samples and {} labels.'
                             .format(X.shape[0], len(y)))

        if self.verbose_:
            tprint('Fitting MLP ensemble with {} regressors'
                   .format(self.n_regressors_))

        self._create_models(X, y)

        if self.backend_== 'sklearn':
            [ model.fit(X, y) for model in self.models_ ]

        elif self.backend_ == 'keras':
            [ model.fit(X, y,
                        batch_size=self.batch_sizes_[model_idx],
                        epochs=self.max_iters_[model_idx],
                        verbose=self.verbose_)
              for model_idx, model in enumerate(self.models_) ]

        if self.verbose_:
            tprint('Done fitting MLP ensemble.')

        return self

    def predict(self, X):
        pred = np.array([ model.predict(X) for model in self.models_ ])
        assert(pred.shape[0] == self.n_regressors_)
        assert(pred.shape[1] == X.shape[0])

        if self.loss_ == 'gaussian_nll':
            assert(pred.shape[2] == 2)

            pred_mean = pred[:, :, 0]
            pred_var = np.exp(pred[:, :, 1])

            ys = pred_mean.mean(0)
            self.uncertainties_ = (
                pred_var + np.power(pred_mean, 2)
            ).mean(0) - np.power(ys, 2)
            self.multi_predict_ = pred_mean.T

            return ys

        elif self.loss_ == 'mse':
            assert(pred.shape[2] == 1)

            self.uncertainties_ = pred.var(0).flatten()
            self.multi_predict_ = pred[:, :, 0].T
            return pred.mean(0).flatten()
