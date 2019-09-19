import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from gaussian_process import SparseGPRegressor
from hybrid import HybridMLPEnsembleGP
from process_davis2011kinase import process, visualize_heatmap
from train_davis2011kinase import train
from utils import tprint

def acquisition_rank(y_pred, var_pred, beta=1.):
    beta = 100. ** (beta - 1.)
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

def select_candidates(explore=False, **kwargs):
    regressor = kwargs['regressor']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    n_candidates = kwargs['n_candidates']

    y_unk_pred = regressor.predict(X_unk)
    var_unk_pred = regressor.uncertainties_

    if explore:
        tprint('Exploring...')
        max_acqs = sorted(set([
            np.argmax(acquisition_rank(y_unk_pred, var_unk_pred, cand))
            for cand in range(1, n_candidates + 1)
        ]))

    else:
        tprint('Exploiting...')
        acquisition = acquisition_rank(y_unk_pred, var_unk_pred)
        max_acqs = np.argsort(-acquisition)[:n_candidates]

    for max_acq in max_acqs:
        tprint('\tAcquire element {} with real Kd value {}'
               .format(idx_unk[max_acq], y_unk[max_acq]))

    return list(max_acqs)

def select_candidates_per_quadrant(**kwargs):
    regressor = kwargs['regressor']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    n_candidates = kwargs['n_candidates']

    acquired = []

    quad_names = [ 'side', 'repurpose', 'novel' ]

    orig_idx = np.array(list(range(X_unk.shape[0])))

    for quad_name in quad_names:
        tprint('Considering quadrant {}'.format(quad_name))

        quad = [ i for i, idx in enumerate(idx_unk)
                 if idx in set(kwargs['idx_' + quad_name]) ]

        idx_unk_quad = [ idx for i, idx in enumerate(idx_unk)
                         if idx in set(kwargs['idx_' + quad_name]) ]

        y_unk_pred = regressor.predict(X_unk[quad])
        var_unk_pred = regressor.uncertainties_
        acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

        max_acqs = np.argsort(-acquisition)[:n_candidates]

        for max_acq in max_acqs:
            tprint('\tAcquire element {} with real Kd value {}'
                   .format(idx_unk_quad[max_acq], y_unk[quad][max_acq]))

        acquired += list(orig_idx[quad][max_acqs])

    return acquired

def select_candidates_per_protein(**kwargs):
    regressor = kwargs['regressor']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    protss = kwargs['prots']

    acquired = []

    orig_idx = np.array(list(range(X_unk.shape)))

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

        acquired.append(orig_idx[involves_prot][max_acq])

    return acquired

def iterate(**kwargs):
    prots = kwargs['prots']
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']
    idx_obs = kwargs['idx_obs']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']

    regressor = kwargs['regressor']
    regress_type = kwargs['regress_type']

    if 'scheme' in kwargs:
        scheme = kwargs['scheme']
    else:
        scheme = 'exploit'
    if 'n_candidates' in kwargs:
        n_candidates = kwargs['n_candidates']
    else:
        kwargs['n_candidates'] = 1

    if scheme == 'exploit':
        acquired = select_candidates(**kwargs)

    elif scheme == 'explore':
        acquired = select_candidates(explore=True, **kwargs)

    elif scheme == 'quad':
        acquired = select_candidates_per_quadrant(**kwargs)

    elif scheme == 'per_prot':
        acquired = select_candidates_per_protein(**kwargs)

    # Reset observations.

    X_acquired = X_unk[acquired]
    y_acquired = y_unk[acquired]

    X_obs = np.vstack((X_obs, X_acquired))
    y_obs = np.hstack((y_obs, y_acquired))
    [ idx_obs.append(idx_unk[a]) for a in acquired ]

    # Reset unknowns.

    unacquired = [ i for i in range(X_unk.shape[0]) if i not in set(acquired) ]

    X_unk = X_unk[unacquired]
    y_unk = y_unk[unacquired]
    idx_unk = [ idx for i, idx in enumerate(idx_unk) if i not in set(acquired) ]

    kwargs['X_obs'] = X_obs
    kwargs['y_obs'] = y_obs
    kwargs['idx_obs'] = idx_obs
    kwargs['X_unk'] = X_unk
    kwargs['y_unk'] = y_unk
    kwargs['idx_unk'] = idx_unk

    return kwargs

if __name__ == '__main__':
    #debug_selection('hybrid')

    param_dict = process()

    param_dict['regress_type'] = 'gp'
    param_dict['scheme'] = 'explore'
    param_dict['n_candidates'] = 10

    for i in range(30):
        tprint('Iteration {}'.format(i))

        param_dict = train(**param_dict)

        param_dict = iterate(**param_dict)
