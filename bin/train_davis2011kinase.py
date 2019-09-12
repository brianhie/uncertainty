import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from gaussian_process import SparseGPRegressor
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

def error_histogram(y_pred, y, regress_type, prefix=''):
    # Histogram of squared errors.
    plt.figure()
    plt.hist(np.power(y_pred - y, 2), bins=50)
    plt.xlabel('Squared Error')
    plt.savefig('figures/mse_histogram_{}regressors{}.png'
                .format(prefix, regress_type))
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

def error_var_scatter(y_pred, y, var_pred, regress_type, prefix=''):
    # Plot error vs. variance.
    plt.figure()
    plt.scatter(y_pred, var_pred, alpha=0.3, c=y)
    plt.viridis()
    #plt.yscale('log')
    plt.xlabel('y_pred')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_pred_{}regressors{}.png'
                .format(prefix, regress_type))
    plt.close()

    plt.figure()
    plt.scatter(y_pred - y, var_pred, alpha=0.3, c=y)
    plt.viridis()
    #plt.yscale('log')
    plt.xlabel('y_pred - y_true')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_error_{}regressors{}.png'
                .format(prefix, regress_type))
    plt.close()

    plt.figure()
    plt.scatter(y, y_pred, alpha=0.3)
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.savefig('figures/true_vs_pred_{}regressors{}.png'
                .format(prefix, regress_type))
    plt.close()


def score_var_scatter(y, var, regress_type, prefix=''):
    # Plot error vs. variance.
    plt.figure()
    plt.scatter(y, var, alpha=0.2)
    #plt.yscale('log')
    plt.xlabel('Predicted Kd')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_score_{}regressors{}.png'
                .format(prefix, regress_type))
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

    n_chems, n_prots = Kds.shape

    regress_type = 'sparsegp'

    if regress_type == 'diverse1':
        regressor = mlp_ensemble_diverse1()
    elif regress_type == 'mlper5':
        regressor = mlp_ensemble(n_neurons=200, regress_type=5)
    else:
        regressor = SparseGPRegressor(n_inducing=100)
    regressor.fit(X_obs, y_obs)

    # Analyze observed dataset.

    y_obs_pred = regressor.predict(X_obs)
    var_obs_pred = regressor.uncertainties_
    assert(len(y_obs_pred) == len(var_obs_pred) == X_obs.shape[0])

    error_histogram(y_obs_pred, y_obs,
                    regress_type, 'observed_')
    error_var_scatter(y_obs_pred, y_obs, var_obs_pred,
                      regress_type, 'observed_')
    score_var_scatter(y_obs, var_obs_pred,
                     regress_type, 'observed_')

    # Analyze unknown dataset.

    y_unk_pred = regressor.predict(X_unk)
    var_unk_pred = regressor.uncertainties_
    assert(len(y_unk_pred) == len(var_unk_pred) == X_unk.shape[0])

    print(sorted(set(var_unk_pred)))

    error_histogram(y_unk_pred, y_unk,
                    regress_type, 'unknown_')
    error_var_scatter(y_unk_pred, y_unk, var_unk_pred,
                      regress_type, 'unknown_')
    score_var_scatter(y_unk, var_unk_pred,
                     regress_type, 'unknown_')

if __name__ == '__main__':
    train(**process())
