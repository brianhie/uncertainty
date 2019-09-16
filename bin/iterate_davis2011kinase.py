import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from gaussian_process import SparseGPRegressor
from hybrid import HybridMLPEnsembleGP
from process_davis2011kinase import process, visualize_heatmap
from train_davis2011kinase import train
from utils import tprint

def acquisition_rank(y_pred, var_pred):
    return rankdata(y_pred) + rankdata(-var_pred)

def acquisition_ucb(y_pred, var_pred, beta=1):
    return y_pred - (beta * var_pred)

def debug_selection(regress_type='gp'):#, **kwargs):
    y_unk_pred = np.loadtxt('target/ypred_unknown_regressors{}.txt'
                            .format(regress_type))
    var_unk_pred = np.loadtxt('target/variance_unknown_regressors{}.txt'
                              .format(regress_type))

    for beta in [ 'rank', 100000, 500000, 1000000, ]:
        if beta == 'rank':
            acquisition = acquisition_rank(y_unk_pred, var_unk_pred)
        else:
            acquisition = acquisition_ucb(y_unk_pred, var_unk_pred, beta=beta)
        plt.figure()
        plt.scatter(y_unk_pred, var_unk_pred, alpha=0.3, c=acquisition)
        plt.viridis()
        plt.title(regress_type.title())
        plt.xlabel('Predicted score')
        plt.ylabel('Variance')
        plt.savefig('figures/acquisition_unknown_regressors{}_beta{}.png'
                    .format(regress_type, beta), dpi=200)
        plt.close()

def select_candidates(beta='auto', **kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    idx_obs = kwargs['idx_obs']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    regressor = kwargs['regressor']
    regress_type = kwargs['regress_type']

    y_unk_pred = regressor.predict(X_unk)
    var_unk_pred = regressor.uncertainties_

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

    max_acq = np.argmax(acquisition)

    tprint('Remove element {} with real Kd value {}'
           .format(idx_unk[max_acq], y_unk[max_acq]))

    X_obs = np.vstack((X_obs, X_unk[max_acq].reshape(1, -1)))
    y_obs = np.hstack((y_obs, np.array([ y_unk[max_acq] ])))
    idx_obs.append(idx_unk[max_acq])

    X_unk = np.vstack((X_unk[:max_acq], X_unk[max_acq + 1:]))
    y_unk = np.hstack((y_unk[:max_acq], y_unk[max_acq + 1:]))
    idx_unk.remove(idx_unk[max_acq])

    kwargs['X_obs'] = X_obs
    kwargs['y_obs'] = y_obs
    kwargs['idx_obs'] = idx_obs
    kwargs['X_unk'] = X_unk
    kwargs['y_unk'] = y_unk
    kwargs['idx_unk'] = idx_unk

    return kwargs

if __name__ == '__main__':
    #debug_selection()

    param_dict = process()

    param_dict['regress_type'] = 'hybrid'

    for i in range(30):
        tprint('Iteration {}'.format(i))

        param_dict = train(**param_dict)

        param_dict = select_candidates(**param_dict)
