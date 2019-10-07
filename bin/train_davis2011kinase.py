import numpy as np
import scipy.stats as ss
import seaborn as sns

from utils import tprint, plt
from bayesian_neural_network import BayesianNN
from gaussian_process import GPRegressor, SparseGPRegressor
from hybrid import HybridMLPEnsembleGP
from process_davis2011kinase import process, visualize_heatmap

def mlp_ensemble_diverse1():
    from mlp_ensemble import MLPEnsembleRegressor

    layer_sizes_list = [ (500, 500) for _ in range(40) ]

    max_iters = [ 500 for _ in range(40) ]

    mlper = MLPEnsembleRegressor(
        layer_sizes_list,
        activations='relu',
        solvers='adam',
        alphas=0.0001,
        batch_sizes=500,
        max_iters=max_iters,
        momentums=0.9,
        nesterovs_momentums=True,
        backend='keras',
        verbose=True,
    )

    return mlper

def mlp_ensemble(n_neurons=500, n_regressors=5, n_epochs=100,
                 loss='mse'):
    from mlp_ensemble import MLPEnsembleRegressor

    layer_sizes_list = []
    for i in range(n_regressors):
        layer_sizes_list.append((n_neurons, n_neurons))

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
        verbose=True,
    )

    return mlper

def error_histogram(y_pred, y, regress_type, prefix=''):
    # Histogram of squared errors.
    plt.figure()
    plt.hist(np.power(y_pred - y, 2), bins=50)
    plt.xlabel('Squared Error')
    plt.savefig('figures/mse_histogram_{}regressors{}.png'
                .format(prefix, regress_type), dpi=200)
    plt.close()

def mean_var_heatmap(idx, y_pred, var_pred, Kds, regress_type, prefix=''):
    means = np.zeros(Kds.shape)
    for idx, mean in zip(idx, y_pred):
        means[idx] = mean
    visualize_heatmap(means,
                      '{}mearegress_type{}'.format(prefix, regress_type))
    variances = np.zeros(Kds.shape)
    for idx, variance in zip(idx, var_pred):
        variances[idx] = variance
    visualize_heatmap(variances,
                      '{}variance_regressors{}'.format(prefix, regress_type))

def score_scatter(y_pred, y, var_pred, regress_type, prefix=''):
    plt.figure()
    if var_pred.max() - var_pred.min() == 0:
        var_color = np.ones(len(var_pred))
    else:
        var_color = (var_pred - var_pred.min()) / (var_pred.max() - var_pred.min())
    plt.scatter(y, y_pred, alpha=0.3, c=var_color)
    plt.viridis()
    plt.xlabel('Real score')
    plt.ylabel('Predicted score')
    plt.savefig('figures/pred_vs_true_{}regressors{}.png'
                .format(prefix, regress_type), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(y_pred, var_pred, alpha=0.3,
                c=(y - y.min()) / (y.max() - y.min()))
    plt.viridis()
    plt.xlabel('Predicted score')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_pred_{}regressors{}.png'
                .format(prefix, regress_type), dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(y, var_pred, alpha=0.3,
                c=(y_pred - y_pred.min()) / (y_pred.max() - y_pred.min()))
    plt.viridis()
    plt.xlabel('Real score')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_true_{}regressors{}.png'
                .format(prefix, regress_type), dpi=200)
    plt.close()

    np.savetxt('target/variance_{}regressors{}.txt'
               .format(prefix, regress_type), var_pred)
    np.savetxt('target/ypred_{}regressors{}.txt'
               .format(prefix, regress_type), y_pred)
    np.savetxt('target/ytrue_{}regressors{}.txt'
               .format(prefix, regress_type), y)

def error_print(y_pred, y, namespace):
    tprint('MSE for {}: {}'
           .format(namespace, np.linalg.norm(y_pred - y)))
    tprint('Pearson rho for {}: {}'
           .format(namespace, ss.pearsonr(y_pred, y)))
    tprint('Spearman r for {}: {}'
           .format(namespace, ss.spearmanr(y_pred, y)))

def train(regress_type='hybrid', **kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    Kds = kwargs['Kds']

    kwargs['regress_type'] = regress_type

    #X_obs = X_obs[:10]
    #y_obs = y_obs[:10]

    # Balance the training set.
    #positive_idx = y_obs > 0
    #zero_idx = y_obs == 0
    #if sum(zero_idx) > sum(positive_idx):
    #    balanced_idx = list(sorted(np.hstack((
    #        np.where(positive_idx)[0],
    #        np.random.choice(np.where(zero_idx)[0], sum(positive_idx), replace=False)
    #    ))))
    #    X_obs = X_obs[balanced_idx]
    #    y_obs = y_obs[balanced_idx]

    # Fit the model.

    if regress_type == 'diverse1':
        regressor = mlp_ensemble_diverse1()

    elif regress_type == 'mlper1':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=1,
            n_epochs=150,
            loss='mse',
        )
    elif regress_type == 'mlper1g':
        regressor = mlp_ensemble(
            n_neurons=100,
            n_regressors=1,
            n_epochs=100,
            loss='gaussian_nll',
        )

    elif regress_type == 'mlper5':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=5,
            n_epochs=100,
        )
    elif regress_type == 'mlper5g':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=5,
            n_epochs=100,
            loss='gaussian_nll',
        )

    elif regress_type == 'bayesnn':
        regressor = BayesianNN(
            n_hidden1=200,
            n_hidden2=200,
            n_iter=5000,
            n_posterior_samples=200,
            verbose=True,
        )

    elif regress_type == 'gp':
        regressor = GPRegressor(
            backend='sklearn',
            n_restarts=10,
            n_jobs=30,
            verbose=True
        )
    elif regress_type == 'sparsegp':
        regressor = SparseGPRegressor(
            method='geosketch',
            n_inducing=8000,
            backend='sklearn',
            n_restarts=10,
            n_jobs=30,
            verbose=True
        )

    elif regress_type == 'hybrid':
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=200,
                n_regressors=1,
                n_epochs=50,
            ),
            GPRegressor(
                backend='sklearn',#'gpytorch',
                n_restarts=10,
                n_jobs=30,
                verbose=True,
            ),
        )
    elif regress_type == 'sparsehybrid':
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=200,
                n_regressors=1,
                n_epochs=50,
            ),
            SparseGPRegressor(
                method='geosketch',
                n_inducing=8000,
                backend='sklearn',
                n_restarts=10,
                n_jobs=30,
                verbose=True
            ),
        )

    regressor.fit(X_obs, y_obs)

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

    y_obs_pred = regressor.predict(X_obs)
    var_obs_pred = regressor.uncertainties_
    assert(len(y_obs_pred) == len(var_obs_pred) == X_obs.shape[0])

    error_histogram(y_obs_pred, y_obs,
                    regress_type, 'observed_')
    score_scatter(y_obs_pred, y_obs, var_obs_pred,
                  regress_type, 'observed_')
    error_print(y_obs_pred, y_obs, 'observed')

    # Analyze unknown dataset.

    y_unk_pred = regressor.predict(X_unk)
    var_unk_pred = regressor.uncertainties_
    assert(len(y_unk_pred) == len(var_unk_pred) == X_unk.shape[0])

    error_histogram(y_unk_pred, y_unk,
                    regress_type, 'unknown_')
    score_scatter(y_unk_pred, y_unk, var_unk_pred,
                  regress_type, 'unknown_')
    error_print(y_unk_pred, y_unk, 'unknown_all')

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
    analyze_regressor(**train(regress_type='bayesnn', **process()))
