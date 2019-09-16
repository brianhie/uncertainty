import GPy
from joblib import Parallel, delayed
from math import ceil
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from utils import tprint

def parallel_predict(model, X, batch_num, batch_size):
    mean, var = model.predict(
        X[batch_num*batch_size:(batch_num+1)*batch_size],
        return_std=True
    )
    return mean, var

class SparseGPRegressor(object):
    def __init__(
            self,
            n_inducing=1000,
            n_restarts=0,
            kernel='rbf',
            backend='sklearn',
            batch_size=1000,
            n_jobs=1,
            verbose=False,
    ):
        self.n_inducing_ = n_inducing
        self.n_restarts_ = n_restarts
        self.kernel_ = kernel
        self.backend_ = backend
        self.batch_size_ = batch_size
        self.n_jobs_ = n_jobs
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
                n_restarts_optimizer=self.n_restarts_,
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
            tprint('Done fitting GP model.')

        return self

    def predict(self, X):
        if self.verbose_:
            tprint('Finding GP model predictions on {} data points...'
                   .format(X.shape[0]))

        if self.backend_ == 'sklearn':
            n_batches = int(ceil(float(X.shape[0]) / self.batch_size_))
            results = Parallel(n_jobs=self.n_jobs_)(
                delayed(parallel_predict)(self.model_, X, batch_num, self.batch_size_)
                for batch_num in range(n_batches)
            )
            mean = np.concatenate([ result[0] for result in results ])
            var = np.concatenate([ result[1] for result in results ])

        elif self.backend_ == 'gpy':
            mean, var = self.model_.predict(X, full_cov=False)

        if self.verbose_:
            tprint('Done predicting with GP model.')

        self.uncertainties_ = var.flatten()
        return mean.flatten()
