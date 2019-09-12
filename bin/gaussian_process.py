import GPy

class SparseGPRegressor(object):
    def __init__(
            self,
            n_inducing=100,
            kernel='rbf',
    ):
        self.n_inducing_ = n_inducing
        self.kernel_ = kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape

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
        self.model_.optimize()

    def predict(self, X):
        mean, var = self.model_._raw_predict(X)
        self.uncertainties_ = var
        return mean
