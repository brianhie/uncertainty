from math import ceil
import numpy as np

def check_param_length(param, n_regressors):
    if len(param) != n_regressors:
        raise ValueError('Invalid parameter list length')

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
                 backend='sklearn',
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
            self.max_iters_ = max_iters_
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
                model.compile(loss='mean_squared_error',
                              optimizer=self.solvers_[model_idx])
                self.models_.append(model)

    def fit(self, X, y):
        if self.backend_== 'sklearn':
            [ model.fit(X, y) for model in self.models_ ]

        elif self.backend == 'keras':
            [ model.fit(X, y,
                        batch_size=self.batch_sizes_[model_idx],
                        epochs=int(ceil(self.max_iters_[model_idx] /
                                        self.batch_sizes_[model_idx])),
                        verbose=self.verbose_)
              for model_idx, model in enumerate(self.models_) ]

        return self

    def predict(self, X):
        ys = np.array([ model.predict(X) for model in self.models_ ])
        self.uncertainties_ = ys.var(0)
        return ys.mean(0)
