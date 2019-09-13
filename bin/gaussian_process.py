import GPy
from sklearn.gaussian_process import GaussianProcessRegressor

from utils import tprint

class SparseGPRegressor(object):
    def __init__(
            self,
            n_inducing=1000,
            kernel='rbf',
            backend='sklearn',
            verbose=False,
    ):
        self.n_inducing_ = n_inducing
        self.kernel_ = kernel
        self.backend_ = backend
        self.verbose_ = verbose

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.verbose_:
            tprint('Fitting GP model on {} data points with dimension {}...'
                   .format(*X.shape))

        # scikit-learn backend.
        if self.backend_ == 'sklearn':
            self.model_ = GaussianProcessRegressor(
                kernel=None,
                normalize_y=True,
                n_restarts_optimizer=1,
                copy_X_train=False,
            ).fit(X, y)

        # GPy backend.
        elif self.backend_ == 'gpy':
            if self.kernel_ == 'rbf':
                kernel = GPy.kern.RBF(
                    input_dim=n_features, variance=1., lengthscale=1.
                )
            else:
                raise ValueError('Kernel value {} not supported'
                                 .format(self.kernel_))

            self.model_ = GPy.models.SparseGPRegression(
                X, y.reshape(-1, 1), kernel=kernel,
                num_inducing=min(self.n_inducing_, n_samples)
            )
            self.model_.Z.unconstrain()
            self.model_.optimize(messages=self.verbose_)

        if self.verbose_:
            tprint('Done fitting model.')

        return self

    def predict(self, X):
        if self.verbose_:
            tprint('Finding GP model predictions on {} data points...'
                   .format(X.shape[0]))

        if self.backend_ == 'sklearn':
            mean, var = self.model_.predict(X, return_std=True)

        elif self.backend_ == 'gpy':
            mean, var = self.model_.predict(X, full_cov=False)

        if self.verbose_:
            tprint('Done predicting.')

        self.uncertainties_ = var.flatten()
        return mean.flatten()
