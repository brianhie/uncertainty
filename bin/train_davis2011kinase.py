import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mlp_ensemble import MLPEnsembleRegressor
from process_davis2011kinase import process, visualize_heatmap

def train(**kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    idx_obs = kwargs['idx_obs']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    Kds = kwargs['Kds']

    n_chems, n_prots = Kds.shape

    n_regressors = 20

    layer_sizes_list = []
    for i in range(n_regressors):
        layer_sizes_list.append((200, 200))

    mlper = MLPEnsembleRegressor(
        layer_sizes_list,
        activations='relu',
        solvers='adam',
        alphas=0.0001,
        batch_sizes=1000,
        max_iters=1000,
        momentums=0.9,
        nesterovs_momentums=True,
        backend='keras',
        verbose=True,
    )
    mlper.fit(X_obs, y_obs)

    # Analyze observed dataset.

    y_obs_pred = mlper.predict(X_obs)
    var_obs_pred = mlper.uncertainties_

    means = np.zeros(Kds.shape)
    for idx, mean in zip(idx_obs, y_obs_pred):
        means[idx] = mean
    visualize_heatmap(means, 'observed_mean')
    variances = np.zeros(Kds.shape)
    for idx, variance in zip(idx_obs, var_obs_pred):
        variances[idx] = variance
    visualize_heatmap(variances, 'observed_variance')

    plt.figure()
    plt.scatter(np.linalg.norm(y_obs_pred - y_obs), var_obs_pred)
    plt.xlabel('MSE')
    plt.ylabel('Variance')
    plt.savefig('figures/variance_vs_mse.png')
    plt.close()

    # Analyze unknown dataset.

    y_unk_pred = mlper.predict(X_unk)
    var_obs_pred = mlper.uncertainties_

    means = np.zeros(Kds.shape)
    for idx, mean in zip(idx_unk, y_unk_pred):
        means[idx] = mean
    visualize_heatmap(means, 'unknown_mean')
    variances = np.zeros(Kds.shape)
    for idx, variance in zip(idx_unk, var_unk_pred):
        variances[idx] = variance
    visualize_heatmap(variances, 'unknown_variance')

if __name__ == '__main__':
    train(**process())
