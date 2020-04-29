from utils import tprint, plt
from process_davis2011kinase import process, visualize_heatmap

import numpy as np
import scipy.stats as ss
import seaborn as sns
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import sys

def mlp_ensemble(n_neurons=500, n_regressors=5, n_epochs=100,
                 n_hidden_layers=2, loss='mse', seed=1,
                 verbose=True,):
    from mlp_ensemble import MLPEnsembleRegressor

    layer_sizes_list = []
    for i in range(n_regressors):
        layer_sizes_list.append((n_neurons,) * n_hidden_layers)

    mlper = MLPEnsembleRegressor(
        layer_sizes_list,
        activations='relu',
        loss=loss,
        solvers='adam',
        alphas=0.1,
        batch_sizes=500,
        max_iters=n_epochs,
        momentums=0.9,
        nesterovs_momentums=True,
        backend='keras',
        random_state=seed,
        verbose=verbose,
    )

    return mlper

def score_scatter(y_pred, y, var_pred, regress_type, prefix=''):
    y_pred = y_pred[:]
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 10000] = 10000

    plt.figure()
    plt.scatter(y_pred, var_pred, alpha=0.3,
                c=(y - y.min()) / (y.max() - y.min()))
    plt.viridis()
    plt.xlabel('Predicted score')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_pred_{}regressors{}.png'
                .format(prefix, regress_type), dpi=300)
    plt.close()

def error_print(y_pred, y, namespace):
    tprint('MAE for {}: {}'
           .format(namespace, mae(y_pred, y)))
    tprint('MSE for {}: {}'
           .format(namespace, mse(y_pred, y)))
    tprint('Pearson rho for {}: {}'
           .format(namespace, ss.pearsonr(y_pred, y)))
    tprint('Spearman r for {}: {}'
           .format(namespace, ss.spearmanr(y_pred, y)))

def train(regress_type='hybrid', seed=1, **kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    Kds = kwargs['Kds']

    kwargs['regress_type'] = regress_type

    # Debug.
    #X_obs = X_obs[:10]
    #y_obs = y_obs[:10]

    # Fit the model.

    if regress_type == 'baseline':
        from baseline import Baseline
        regressor = Baseline()

    elif regress_type == 'mlper1':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=1,
            n_epochs=50,
            seed=seed,
        )
    elif regress_type == 'dmlper1':
        regressor = mlp_ensemble(
            n_neurons=1000,
            n_regressors=1,
            n_hidden_layers=15,
            n_epochs=300,
            seed=seed,
        )
    elif regress_type == 'mlper1g':
        regressor = mlp_ensemble(
            n_neurons=100,
            n_regressors=1,
            n_epochs=100,
            loss='gaussian_nll',
            seed=seed,
        )

    elif regress_type == 'mlper5':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=5,
            n_epochs=100,
            seed=seed,
        )
    elif regress_type == 'mlper5g':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=5,
            n_epochs=50,
            loss='gaussian_nll',
            seed=seed,
        )

    elif regress_type == 'bayesnn':
        from bayesian_neural_network import BayesianNN
        regressor = BayesianNN(
            n_hidden1=200,
            n_hidden2=200,
            n_iter=1000,
            n_posterior_samples=100,
            random_state=seed,
            verbose=True,
        )

    elif regress_type == 'cmf':
        from cmf_regressor import CMFRegressor
        regressor = CMFRegressor(
            n_components=30,
            seed=seed,
        )
        regressor.fit(
            kwargs['chems'],
            kwargs['prots'],
            kwargs['chem2feature'],
            kwargs['prot2feature'],
            kwargs['Kds'],
            kwargs['idx_obs'],
        )

    elif regress_type == 'gp':
        from gaussian_process import GPRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        regressor = GPRegressor(
            kernel=C(10000., 'fixed') * RBF(1., 'fixed'),
            backend='sklearn',
            n_jobs=10,
            verbose=True
        )
    elif regress_type == 'gpfactorized':
        from gaussian_process import GPRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C
        from kernels import FactorizedRBF

        n_features_chem = kwargs['n_features_chem']
        n_features_prot = kwargs['n_features_prot']

        regressor = GPRegressor(
            kernel=C(1., 'fixed') * FactorizedRBF(
                [ 1.1, 1. ], [ n_features_chem, n_features_prot ], 'fixed'
            ),
            backend='sklearn',
            n_restarts=0,
            n_jobs=10,
            verbose=True
        )
    elif regress_type == 'sparsegp':
        from gaussian_process import SparseGPRegressor
        regressor = SparseGPRegressor(
            method='geosketch',
            n_inducing=8000,
            backend='sklearn',
            n_restarts=10,
            n_jobs=10,
            verbose=True
        )

    elif regress_type == 'hybrid':
        from gaussian_process import GPRegressor
        from hybrid import HybridMLPEnsembleGP
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=200,
                n_regressors=1,
                n_epochs=50,
                seed=seed,
            ),
            GPRegressor(
                backend='sklearn',#'gpytorch',
                n_restarts=10,
                n_jobs=10,
                verbose=True,
            ),
        )
    elif regress_type == 'dhybrid':
        from gaussian_process import GPRegressor
        from hybrid import HybridMLPEnsembleGP
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=1000,
                n_regressors=1,
                n_hidden_layers=15,
                n_epochs=300,
                seed=seed,
            ),
            GPRegressor(
                backend='sklearn',#'gpytorch',
                n_restarts=10,
                n_jobs=10,
                verbose=True,
            ),
        )
    elif regress_type == 'sparsehybrid':
        from gaussian_process import SparseGPRegressor
        from hybrid import HybridMLPEnsembleGP
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=200,
                n_regressors=1,
                n_epochs=50,
                seed=seed,
            ),
            SparseGPRegressor(
                method='geosketch',
                n_inducing=8000,
                backend='sklearn',
                n_restarts=10,
                n_jobs=10,
                verbose=True
            ),
        )

    if regress_type not in { 'cmf' }:
        regressor.fit(X_obs, y_obs)

    #print(regressor.model_.kernel_.get_params()) # Debug.

    kwargs['regressor'] = regressor

    return kwargs

def analyze_regressor(**kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    regressor = kwargs['regressor']
    regress_type = kwargs['regress_type']

    # Analyze observed dataset.

    if regress_type == 'cmf':
        y_obs_pred = regressor.predict(kwargs['idx_obs'])
    else:
        y_obs_pred = regressor.predict(X_obs)
    var_obs_pred = regressor.uncertainties_
    assert(len(y_obs_pred) == len(var_obs_pred) == X_obs.shape[0])

    error_print(y_obs_pred, y_obs, 'observed')
    score_scatter(y_obs_pred, y_obs, var_obs_pred,
                  regress_type, 'observed_')

    # Analyze unknown dataset.

    if regress_type == 'cmf':
        y_unk_pred = regressor.predict(kwargs['idx_unk'])
    else:
        y_unk_pred = regressor.predict(X_unk)
    var_unk_pred = regressor.uncertainties_
    assert(len(y_unk_pred) == len(var_unk_pred) == X_unk.shape[0])

    error_print(y_unk_pred, y_unk, 'unknown_all')
    score_scatter(y_unk_pred, y_unk, var_unk_pred,
                  regress_type, 'unknown_')

    # Stratify unknown dataset into quadrants.

    idx_side = [ i for i, idx in enumerate(idx_unk)
                 if idx in set(kwargs['idx_side']) ]
    idx_repurpose = [ i for i, idx in enumerate(idx_unk)
                      if idx in set(kwargs['idx_repurpose']) ]
    idx_novel = [ i for i, idx in enumerate(idx_unk)
                  if idx in set(kwargs['idx_novel']) ]
    error_print(y_unk_pred[idx_side], y_unk[idx_side],
                'unknown_side')
    error_print(y_unk_pred[idx_repurpose], y_unk[idx_repurpose],
                'unknown_repurpose')
    error_print(y_unk_pred[idx_novel], y_unk[idx_novel],
                'unknown_novel')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('regress_type', help='model to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    analyze_regressor(**train(
        regress_type=args.regress_type,
        seed=args.seed,
        **process()
    ))
