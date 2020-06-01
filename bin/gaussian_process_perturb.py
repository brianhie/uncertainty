import gpytorch
from joblib import Parallel, delayed
from math import ceil
import numpy as np
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
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=y.shape[1],
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_constraint=gpytorch.constraints.Interval(1e-5, 1e3),
                    #lengthscale_prior=gpytorch.priors.NormalPrior(30, 20),
                ),
                outputscale_prior=gpytorch.priors.NormalPrior(10, 5),
                outputscale_constraint=gpytorch.constraints.Interval(1e-5, 1e3),
            ),
            num_tasks=y.shape[1], rank=1,
        )

    def forward(self, X):
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return gpytorch.distributions.MultitaskMultivariateNormal(
            mean_X, covar_X
        )

class GPRegressor(object):
    def __init__(
            self,
            n_restarts=0,
            kernel=None,
            backend='sklearn',
            batch_size=1000,
            n_jobs=1,
            seed=None,
            verbose=False,
    ):
        if seed is not None:
            np.random.seed(seed)
            if backend == 'gpytorch':
                import torch
                torch.manual_seed(seed)

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
                kernel=self.kernel_,
                normalize_y=True,
                n_restarts_optimizer=self.n_restarts_,
                copy_X_train=False,
            ).fit(X, y)
            if self.verbose_:
                tprint('Kernel: {}'.format(self.model_.kernel_))
                tprint('Kernel theta: {}'.format(self.model_.kernel_.theta))

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
            n_tasks = y.shape[1]

            X = torch.Tensor(X).contiguous().cuda()
            y = torch.Tensor(y).contiguous().cuda()

            lowest_loss = float('inf')

            for restart in range(self.n_restarts_):
                if restart > 0 and self.verbose_ > 1:
                    tprint('GP optimization restart {}'.format(restart))

                likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                    num_tasks=n_tasks).cuda()
                model = GPyTorchRegressor(X, y, likelihood).cuda()

                model.train()
                likelihood.train()

                # Use the Adam optimizer.
                optimizer = torch.optim.Adam([
                    { 'params': model.parameters() },
                ], lr=1.)

                # Loss for GPs is the marginal log likelihood.
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    likelihood, model
                )

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    training_iterations = 200
                    for i in range(training_iterations):
                        optimizer.zero_grad()
                        output = model(X)
                        loss = -mll(output, y)
                        loss.backward()
                        if self.verbose_ > 1 and \
                           (i % 100 == 0 or i == training_iterations - 1):
                            tprint('Iter {}/{} - Loss: {:.3f}'
                                   .format(i + 1, training_iterations,
                                           loss.item()))
                        optimizer.step()

                if restart == 0 or loss.item() < lowest_loss:
                    self.model_ = model
                    self.likelihood_ = likelihood
                    lowest_loss = loss.item()

        if self.verbose_:
            tprint('Done fitting GP model.')

        return self

    def predict(self, X):
        if self.verbose_:
            tprint('Finding GP model predictions on {} data points...'
                   .format(X.shape[0]))

        if self.backend_ == 'sklearn':
            n_batches = int(ceil(float(X.shape[0]) / self.batch_size_))
            #results = Parallel(n_jobs=self.n_jobs_)(#, max_nbytes=None)(
            #    delayed(parallel_predict)(
            #        self.model_,
            #        X[batch_num*self.batch_size_:(batch_num+1)*self.batch_size_],
            #        batch_num, n_batches, self.verbose_
            #    )
            #    for batch_num in range(n_batches)
            #)
            results = [
                parallel_predict(
                    self.model_,
                    X[batch_num*self.batch_size_:(batch_num+1)*self.batch_size_],
                    batch_num, n_batches, self.verbose_
                )
                for batch_num in range(n_batches)
            ]
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
                 gpytorch.settings.fast_pred_var():#, \
                 #gpytorch.settings.max_root_decomposition_size(35):
                preds = self.model_(X)

            mean = preds.mean.detach().cpu().numpy()
            var = preds.variance.detach().cpu().numpy()
            var = var.mean(1)

        if self.verbose_:
            tprint('Done predicting with GP model.')

        self.uncertainties_ = var
        return mean

class SparseGPRegressor(object):
    def __init__(
            self,
            n_inducing=1000,
            method='geosketch',
            n_restarts=0,
            kernel=None,
            backend='sklearn',
            batch_size=1000,
            seed=None,
            n_jobs=1,
            verbose=False,
    ):
        if seed is not None:
            np.random.seed(seed)
            if backend == 'gpytorch':
                import torch
                torch.manual_seed(seed)

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

                if X.shape[1] > 100:
                    U, s, _ = pca(X, k=100)
                    X_dimred = U * s
                else:
                    X_dimred = X
                gs_idx = gs(X_dimred, self.n_inducing_, replace=False)
                X_sketch = X[gs_idx]
                y_sketch = y[gs_idx]

            else:
                raise ValueError('Invalid sketching method {}'
                                 .format(self.method_))

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


    def predict(self, X, return_std=False):
        y_pred = self.gpr_.predict(X)
        self.uncertainties_ = self.gpr_.uncertainties_
        if return_std:
            return y_pred, self.uncertainties_
        else:
            return y_pred
