import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mlp_ensemble import MLPEnsembleRegressor
from process_davis2011kinase import process, visualize_heatmap

def mlp_ensemble_diverse1():
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

def mlp_ensemble(n_neurons=500, n_regressors=5):
    layer_sizes_list = []
    for i in range(n_regressors):
        layer_sizes_list.append((500, 500))

    mlper = MLPEnsembleRegressor(
        layer_sizes_list,
        activations='relu',
        solvers='adam',
        alphas=0.0001,
        batch_sizes=500,
        max_iters=8000,
        momentums=0.9,
        nesterovs_momentums=True,
        backend='keras',
        verbose=True,
    )

    return mlper

def error_histogram(y_pred, y, n_regressors, prefix=''):
    # Histogram of squared errors.
    plt.figure()
    plt.hist(np.power(y_pred - y, 2), bins=50)
    plt.xlabel('Squared Error')
    plt.savefig('figures/mse_histogram_{}regressors{}.png'
                .format(prefix, n_regressors))
    plt.close()

def mean_var_heatmap(idx, y_pred, var_pred, Kds, n_regressors, prefix=''):
    means = np.zeros(Kds.shape)
    for idx, mean in zip(idx, y_pred):
        means[idx] = mean
    visualize_heatmap(means,
                      '{}mean_regressors{}'.format(prefix, n_regressors))
    variances = np.zeros(Kds.shape)
    for idx, variance in zip(idx, var_pred):
        variances[idx] = variance
    visualize_heatmap(variances,
                      '{}variance_regressors{}'.format(prefix, n_regressors))

def error_var_scatter(y_pred, y, var_pred, n_regressors, prefix=''):
    # Plot error vs. variance.
    plt.figure()
    plt.scatter(np.power(y_pred - y, 2), var_pred, alpha=0.3,
                c=((y - y.min()) / (y.max() - y.min())))
    plt.viridis()
    plt.yscale('log')
    plt.xlabel('Squared Error')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_mse_{}regressors{}.png'
                .format(prefix, n_regressors))
    plt.close()

def score_var_scatter(y, var, n_regressors, prefix=''):
    # Plot error vs. variance.
    plt.figure()
    plt.scatter(y, var, alpha=0.2)
    plt.yscale('log')
    plt.xlabel('Predicted Kd')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_score_{}regressors{}.png'
                .format(prefix, n_regressors))
    plt.close()

def train(**kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    idx_obs = kwargs['idx_obs']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    Kds = kwargs['Kds']

    X_obs = X_obs[y_obs > 0]
    y_obs = y_obs[y_obs > 0]
    X_unk = X_unk[y_unk > 0]
    y_unk = y_unk[y_unk > 0]

    n_chems, n_prots = Kds.shape

    n_regressors = 5

    if n_regressors == 'diverse1':
        mlper = mlp_ensemble_diverse1()
    else:
        mlper = mlp_ensemble(n_neurons=500, n_regressors=n_regressors)
    mlper.fit(X_obs, y_obs)

    # Analyze observed dataset.

    y_obs_pred = mlper.predict(X_obs)
    var_obs_pred = mlper.uncertainties_
    assert(len(y_obs_pred) == len(var_obs_pred) == X_obs.shape[0])

    error_histogram(y_obs_pred, y_obs,
                    n_regressors, 'observed_')
    error_var_scatter(y_obs_pred, y_obs, var_obs_pred,
                      n_regressors, 'observed_')
    score_var_scatter(y_obs, var_obs_pred,
                     n_regressors, 'observed_')

    # Analyze unknown dataset.

    y_unk_pred = mlper.predict(X_unk)
    var_unk_pred = mlper.uncertainties_
    assert(len(y_unk_pred) == len(var_unk_pred) == X_unk.shape[0])

    error_histogram(y_unk_pred, y_unk,
                    n_regressors, 'unknown_')
    error_var_scatter(y_unk_pred, y_unk, var_unk_pred,
                      n_regressors, 'unknown_')
    score_var_scatter(y_unk, var_unk_pred,
                     n_regressors, 'unknown_')

if __name__ == '__main__':
    train(**process())
