import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from gaussian_process import SparseGPRegressor
from hybrid import HybridMLPEnsembleGP
from process_davis2011kinase import process, visualize_heatmap
from train_davis2011kinase import train
from utils import tprint

def acquisition_rank(y_pred, var_pred, beta=1.):
    return rankdata(y_pred) + ((1. / beta) * rankdata(-var_pred))

def acquisition_ucb(y_pred, var_pred, beta=1):
    return y_pred - (beta * var_pred)

def debug_selection(regress_type='gp'):#, **kwargs):
    y_obs_pred = np.loadtxt('target/ytrue_unknown_regressors{}.txt'
                            .format(regress_type))
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

    for beta in range(1, 11):
        acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta=beta)
        print('beta: {}, Kd: {}'.format(beta, y_obs_pred[np.argmax(acquisition)]))

    exit()

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

    max_acqs = np.argsort(-acquisition)[:10]

    max_acq = max_acqs[0]

    tprint('Remove element {} with real Kd value {}'
           .format(idx_unk[max_acq], y_unk[max_acq]))
    tprint('Top acquisition real Kds were {}'
           .format(', '.join([ str(y) for y in y_unk[max_acqs] ])))

    X_obs = np.vstack((X_obs, X_unk[max_acq].reshape(1, -1)))
    #y_obs = np.hstack((y_obs, np.array([ 0. ])))
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

def select_candidates_per_quadrant(beta='auto', **kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    idx_obs = kwargs['idx_obs']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    regressor = kwargs['regressor']
    regress_type = kwargs['regress_type']

    quad_names = [ 'side', 'repurpose', 'novel' ]

    to_remove = set()

    for quad_name in quad_names:
        tprint('Considering quadrant {}'.format(quad_name))

        quad = [ i for i, idx in enumerate(idx_unk)
                 if idx in set(kwargs['idx_' + quad_name]) ]

        idx_unk_quad = [ idx for i, idx in enumerate(idx_unk)
                         if idx in set(kwargs['idx_' + quad_name]) ]

        y_unk_pred = regressor.predict(X_unk[quad])
        var_unk_pred = regressor.uncertainties_
        acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

        max_acqs = np.argsort(-acquisition)[:10]

        for max_acq in max_acqs:
            tprint('\tRemove element {} with real Kd value {}'
                   .format(idx_unk_quad[max_acq], y_unk[quad][max_acq]))

    return kwargs

def select_candidates_per_protein(beta='auto', **kwargs):
    prots = kwargs['prots']
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    idx_obs = kwargs['idx_obs']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    regressor = kwargs['regressor']
    regress_type = kwargs['regress_type']

    for prot_idx, prot in enumerate(prots):
        involves_prot = [ j == prot_idx for i, j in idx_unk ]
        X_unk_prot = X_unk[involves_prot]
        y_unk_prot = y_unk[involves_prot]

        y_unk_pred = regressor.predict(X_unk_prot)
        var_unk_pred = regressor.uncertainties_

        acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

        max_acq = np.argmax(acquisition)

        tprint('Protein {} ({}) has Kd {}'
               .format(prot_idx, prot, y_unk_prot[max_acq]))


if __name__ == '__main__':
    #debug_selection('hybrid')

    param_dict = process()

    param_dict['regress_type'] = 'hybrid'

    #select_candidates_per_protein(**train(**param_dict))
    #exit()

    select_candidates_per_quadrant(**train(**param_dict))
    exit()

    for i in range(30):
        tprint('Iteration {}'.format(i))

        param_dict = train(**param_dict)

        param_dict = select_candidates(**param_dict)
