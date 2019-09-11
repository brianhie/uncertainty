from math import ceil
import numpy as np
import tensorflow as tf
import keras.backend as K

def check_param_length(param, n_regressors):
    if len(param) != n_regressors:
        raise ValueError('Invalid parameter list length')

def gaussian_nll(y_true, y_pred):
    n_dims = int(int(y_pred.shape[1]) / 2.)
    mu = y_pred[:, :n_dims]
    log_sigma = y_pred[:, n_dims:]

    mse = -0.5 * K.sum(K.square((y_true - mu) / K.exp(log_sigma)), axis=1)
    sigma_trace = -K.sum(log_sigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)

    log_likelihood = mse + sigma_trace + log2pi

    return K.mean(-log_likelihood)

class MLPEnsembleRegressor(object):
    def __init__(self,
                 layer_sizes_list,
                 activations='relu',
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
            from keras.models import Sequential
            from keras.layers import Dense

            for model_idx in range(self.n_regressors_):
                hidden_layer_sizes = self.layer_sizes_list_[model_idx]

                model = Sequential()
                for layer_size in hidden_layer_sizes:
                    model.add(Dense(layer_size, kernel_initializer='normal',
                                    activation=self.activations_[model_idx]))
                model.add(
                    Dense(2, kernel_initializer='normal',
                          activation=self.activations_[model_idx])
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
            print('Fitting MLP ensemble with {} regressors'
                  .format(self.n_regressors_))

        self._create_models(X, y)

        if self.backend_== 'sklearn':
            [ model.fit(X, y) for model in self.models_ ]

        elif self.backend_ == 'keras':
            [ model.fit(X, y,
                        batch_size=self.batch_sizes_[model_idx],
                        epochs=int(ceil(self.batch_sizes_[model_idx] *
                                        self.max_iters_[model_idx] /
                                        X.shape[0])),
                        verbose=self.verbose_)
              for model_idx, model in enumerate(self.models_) ]

        return self

    def predict(self, X):
        if self.backend_ == 'sklearn':
            ys = np.array([ model.predict(X) for model in self.models_ ])
            self.uncertainties_ = ys.var(0).flatten()
            return ys.mean(0).flatten()

        elif self.backend_ == 'keras':
            pred = np.array([ model.predict(X) for model in self.models_ ])
            assert(pred.shape[0] == self.n_regressors_)
            assert(pred.shape[1] == X.shape[0])
            assert(pred.shape[2] == 2)

            pred_mean = pred[:, :, 0]
            pred_sdev = np.exp(pred[:, :, 1])

            ys = pred_mean.mean(0)
            self.uncertainties_ = (
                pred_sdev + np.power(pred_mean, 2)
            ).mean(0) - np.power(ys, 2)

            return ys
