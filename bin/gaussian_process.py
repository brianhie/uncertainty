import gpytorch
from joblib import Parallel, delayed
from math import ceil
from sklearn.gaussian_process import GaussianProcessRegressor
import torch

from utils import *

def parallel_predict(model, X, batch_num, n_batches, verbose):
    mean, var = model.predict(X, return_std=True)
    if verbose:
        tprint('Finished predicting batch number {}/{}'
               .format(batch_num + 1, n_batches))
    return mean, var

class GPyTorchRegressor(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super(GPyTorchRegressor, self).__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, X):
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean_X, covar_X)

class GPRegressor(object):
    def __init__(
            self,
            n_restarts=0,
            kernel=None,
            normalize_y=True,
            backend='sklearn',
            batch_size=1000,
            n_jobs=1,
            verbose=False,
    ):
        self.n_restarts_ = n_restarts
        self.kernel_ = kernel
        self.normalize_y_ = normalize_y
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
                kernel=self.kernel_,
                normalize_y=self.normalize_y_,
                n_restarts_optimizer=self.n_restarts_,
                copy_X_train=False,
            ).fit(X, y)

        # GPy backend.
        elif self.backend_ == 'gpy':
            import GPy
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

        # GPyTorch with CUDA backend.
        elif self.backend_ == 'gpytorch':
            X = torch.Tensor(X).contiguous().cuda()
            y = torch.Tensor(y).contiguous().cuda()

            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            model = GPyTorchRegressor(X, y, likelihood).cuda()

            model.train()
            likelihood.train()

            # Use the Adam optimizer.
            #optimizer = torch.optim.LBFGS([ {'params': model.parameters()} ])
            optimizer = torch.optim.Adam([
                {'params': model.parameters()}, # Includes GaussianLikelihood parameters.
            ], lr=1.)

            # Loss for GPs is the marginal log likelihood.
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            training_iterations = 100
            for i in range(training_iterations):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, y)
                loss.backward()
                if self.verbose_:
                    tprint('Iter {}/{} - Loss: {:.3f}'
                           .format(i + 1, training_iterations, loss.item()))
                optimizer.step()

            self.model_ = model
            self.likelihood_ = likelihood

        if self.verbose_:
            tprint('Done fitting GP model.')

        return self

    def predict(self, X):
        if self.verbose_:
            tprint('Finding GP model predictions on {} data points...'
                   .format(X.shape[0]))

        if self.backend_ == 'sklearn':
            n_batches = int(ceil(float(X.shape[0]) / self.batch_size_))
            results = Parallel(n_jobs=self.n_jobs_)(#, max_nbytes=None)(
                delayed(parallel_predict)(
                    self.model_,
                    X[batch_num*self.batch_size_:(batch_num+1)*self.batch_size_],
                    batch_num, n_batches, self.verbose_
                )
                for batch_num in range(n_batches)
            )
            mean = np.concatenate([ result[0] for result in results ])
            var = np.concatenate([ result[1] for result in results ])

        elif self.backend_ == 'gpy':
            mean, var = self.model_.predict(X, full_cov=False)

        elif self.backend_ == 'gpytorch':
            X = torch.Tensor(X).contiguous().cuda()

            # Set into eval mode.
            self.model_.eval()
            self.likelihood_.eval()

            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.max_root_decomposition_size(35):
                preds = self.model_(X)

            mean = preds.mean.detach().cpu().numpy()
            var = preds.variance.detach().cpu().numpy()

        if self.verbose_:
            tprint('Done predicting with GP model.')

        self.uncertainties_ = var.flatten()
        return mean.flatten()

class SparseGPRegressor(object):
    def __init__(
            self,
            n_inducing=1000,
            method='geoskech',
            n_restarts=0,
            kernel=None,
            backend='sklearn',
            batch_size=1000,
            n_jobs=1,
            verbose=False,
    ):
        self.n_inducing_ = n_inducing
        self.method_ = method
        self.n_restarts_ = n_restarts
        self.kernel_ = kernel
        self.backend_ = backend
        self.batch_size_ = batch_size
        self.n_jobs_ = n_jobs
        self.verbose_ = verbose

    def fit(self, X, y):
        if X.shape[0] > self.n_inducing_:
            if self.method_ == 'uniform':
                uni_idx = np.random.choice(X.shape[0], self.n_inducing_,
                                           replace=False)
                X_sketch = X[uni_idx]
                y_sketch = y[uni_idx]

            elif self.method_ == 'geosketch':
                from fbpca import pca
                from geosketch import gs

                U, s, _ = pca(X, k=100)
                X_dimred = U[:, :100] * s[:100]
                gs_idx = gs(X_dimred, self.n_inducing_, replace=False)
                X_sketch = X[gs_idx]
                y_sketch = y[gs_idx]

        else:
            X_sketch, y_sketch = X, y

        self.gpr_ = GPRegressor(
            n_restarts=self.n_restarts_,
            kernel=self.kernel_,
            backend=self.backend_,
            batch_size=self.batch_size_,
            n_jobs=self.n_jobs_,
            verbose=self.verbose_,
        ).fit(X_sketch, y_sketch)


    def predict(self, X):
        y_pred = self.gpr_.predict(X)
        self.uncertainties_ = self.gpr_.uncertainties_
        return y_pred
