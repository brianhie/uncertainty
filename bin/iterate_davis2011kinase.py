import numpy as np
from scipy.stats import rankdata
import sys

from utils import tprint, plt
from gaussian_process import SparseGPRegressor
from hybrid import HybridMLPEnsembleGP
from process_davis2011kinase import process, visualize_heatmap
from train_davis2011kinase import train

def acquisition_rank(y_pred, var_pred, beta=1.):
    beta = 100. ** (beta - 1.)
    return rankdata(y_pred) + ((1. / beta) * rankdata(-var_pred))

def acquisition_ucb(y_pred, var_pred, beta=1.):
    return y_pred - (beta * var_pred)

def acquisition_scatter(y_unk_pred, var_unk_pred, acquisition, regress_type):
    plt.figure()
    plt.scatter(y_unk_pred, var_unk_pred, alpha=0.1, c=acquisition)
    plt.viridis()
    plt.title(regress_type.title())
    plt.xlabel('Predicted score')
    plt.ylabel('Variance')
    plt.savefig('figures/acquisition_unknown_{}.png'
                .format(regress_type), dpi=200)
    plt.close()

def debug_selection(regress_type='gp'):
    y_unk_pred = np.loadtxt('target/ypred_unknown_regressors{}.txt'
                            .format(regress_type))
    var_unk_pred = np.loadtxt('target/variance_unknown_regressors{}.txt'
                              .format(regress_type))

    for beta in [ 'rank', 100000, 500000, 1000000, ]:
        if beta == 'rank':
            acquisition = acquisition_rank(y_unk_pred, var_unk_pred)
        else:
            acquisition = acquisition_ucb(y_unk_pred, var_unk_pred, beta=beta)
        acquisition_scatter(y_unk_pred, var_unk_pred, acquisition, regress_type)

    for beta in range(1, 11):
        acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta=beta)
        print('beta: {}, Kd: {}'.format(beta, y_obs_pred[np.argmax(acquisition)]))

    exit()

def select_candidates(explore=False, **kwargs):
    y_unk_pred = kwargs['y_unk_pred']
    var_unk_pred = kwargs['var_unk_pred']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    n_candidates = kwargs['n_candidates']
    chems = kwargs['chems']
    prots = kwargs['prots']

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
        i, j = idx_unk[max_acq]
        chem = chems[i]
        prot = prots[j]

        if y_unk is None:
            tprint('\tAcquire {} {} <--> {} with predicted Kd value {:.3f}'
                   ' and variance {:.3f}'
                   .format((i, j), chem, prot, y_unk_pred[max_acq],
                           var_unk_pred[max_acq]))
        else:
            tprint('\tAcquire {} {} <--> {} with real Kd value {}'
                   .format((i, j), chem, prot, y_unk[max_acq]))

    return list(max_acqs)

def select_candidates_per_quadrant(explore=False, **kwargs):
    y_unk_pred = kwargs['y_unk_pred']
    var_unk_pred = kwargs['var_unk_pred']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    n_candidates = kwargs['n_candidates']
    chems = kwargs['chems']
    prots = kwargs['prots']

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

    acquired = []

    quad_names = [ 'side', 'repurpose', 'novel' ]

    orig_idx = np.array(list(range(X_unk.shape[0])))

    for quad_name in quad_names:
        if explore:
            tprint('Exploring quadrant {}'.format(quad_name))
        else:
            tprint('Considering quadrant {}'.format(quad_name))

        quad = [ i for i, idx in enumerate(idx_unk)
                 if idx in set(kwargs['idx_' + quad_name]) ]

        y_unk_quad = y_unk_pred[quad]
        var_unk_quad = var_unk_pred[quad]
        idx_unk_quad = [ idx for i, idx in enumerate(idx_unk)
                         if idx in set(kwargs['idx_' + quad_name]) ]

        if explore:
            max_acqs = sorted(set([
                np.argmax(acquisition_rank(y_unk_quad, var_unk_quad, cand))
                for cand in range(1, n_candidates + 1)
            ]))
        else:
            max_acqs = np.argsort(-acquisition[quad])[:n_candidates]

        for max_acq in max_acqs:
            i, j = idx_unk_quad[max_acq]
            chem = chems[i]
            prot = prots[j]

            if y_unk is None:
                tprint('\tAcquire {} {} <--> {} with predicted Kd value {}'
                       .format((i, j), chem, prot, y_unk_quad[max_acq]))
            else:
                tprint('\tAcquire {} {} <--> {} with real Kd value {}'
                       .format((i, j), chem, prot, y_unk[quad][max_acq]))

        acquired += list(orig_idx[quad][max_acqs])

    return acquired

def select_candidates_per_protein(**kwargs):
    y_unk_pred = kwargs['y_unk_pred']
    var_unk_pred = kwargs['var_unk_pred']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    chems = kwargs['chems']
    prots = kwargs['prots']
    n_candidates = kwargs['n_candidates']

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

    acquired = []

    orig_idx = np.array(list(range(X_unk.shape[0])))

    for prot_idx, prot in enumerate(prots):
        involves_prot = [ j == prot_idx for i, j in idx_unk ]
        idx_unk_prot = [ (i, j) for i, j in idx_unk if j == prot_idx ]

        max_acqs = np.argsort(-acquisition[involves_prot])[:n_candidates]

        tprint('Protein {}'.format(prot))

        for max_acq in max_acqs:
            i, j = idx_unk_prot[max_acq]
            chem = chems[i]
            prot = prots[j]

            if y_unk is None:
                tprint('\tAcquire {} {} <--> {} with predicted Kd value {:.3f}'
                       ' and variance {:.3f}'
                       .format((i, j), chem, prot, y_unk_pred[involves_prot][max_acq],
                               var_unk_pred[involves_prot][max_acq]))
            else:
                tprint('\tAcquire {} {} <--> {} with real Kd value {}'
                       .format((i, j), chem, prot, y_unk[involves_prot][max_acq]))

            acquired.append(orig_idx[involves_prot][max_acq])

    return acquired

def select_candidates_per_partition(**kwargs):
    y_unk_pred = kwargs['y_unk_pred']
    var_unk_pred = kwargs['var_unk_pred']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    n_partitions = kwargs['n_candidates']
    chems = kwargs['chems']
    prots = kwargs['prots']
    chem2feature = kwargs['chem2feature']

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

    if 'partition' in kwargs:
        partition = kwargs['partition']

    else:
        # Partition unknown space using k-means on chemicals.

        from sklearn.cluster import KMeans
        labels = KMeans(
            n_clusters=n_partitions,
            init='k-means++',
            n_init=3,
        ).fit_predict(np.array([
            chem2feature[chem] for chem in chems
        ]))

        partition = []
        for p in range(n_partitions):
            partition.append([
                idx for idx, (i, j) in enumerate(idx_unk)
                if labels[i] == p
            ])

    orig2new_idx = { i: i for i in range(X_unk.shape[0]) }

    for pi in range(len(partition)):
        if len(partition[pi]) == 0:
            tprint('Partition {} is empty'.format(pi))
            continue

        partition_pi = set(list(partition[pi]))
        idx_unk_part = [ idx for i, idx in enumerate(idx_unk)
                         if i in partition_pi ]

        max_acq = np.argmax(acquisition[partition[pi]])

        i, j = idx_unk_part[max_acq]
        chem = chems[i]
        prot = prots[j]

        tprint('Partition {}'.format(pi))
        if y_unk is None:
            tprint('\tAcquire {} {} <--> {} with predicted Kd value {:.3f}'
                   ' and variance {:.3f}'
                   .format((i, j), chem, prot, y_unk_pred[partition[pi]][max_acq],
                           var_unk_pred[partition[pi]][max_acq]))
        else:
            tprint('\tAcquire {} {} <--> {} with real Kd value {}'
                   .format((i, j), chem, prot, y_unk[partition[pi]][max_acq]))

        orig_max_acq = partition[pi][max_acq]
        for i in orig2new_idx:
            if i == orig_max_acq:
                orig2new_idx[i] = None
            elif orig2new_idx[i] is None:
                pass
            elif i > orig_max_acq:
                orig2new_idx[i] -= 1

    # Acquire one point per partition.

    acquired = sorted([ i for i in orig2new_idx if orig2new_idx[i] is None ])

    # Make sure new partition indices match new unknown dataset.

    for pi in range(len(partition)):
        partition[pi] = np.array([
            orig2new_idx[p] for p in partition[pi]
            if orig2new_idx[p] is not None
        ])

    kwargs['partition'] = partition

    return acquired, kwargs

def acquire(**kwargs):
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

    elif scheme == 'quadexplore':
        acquired = select_candidates_per_quadrant(explore=True, **kwargs)

    elif scheme == 'perprot':
        acquired = select_candidates_per_protein(**kwargs)

    elif scheme == 'partition':
        acquired, kwargs = select_candidates_per_partition(**kwargs)

    return acquired, kwargs

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

    kwargs['y_unk_pred'] = regressor.predict(X_unk)
    kwargs['var_unk_pred'] = regressor.uncertainties_

    acquired, kwargs = acquire(**kwargs)

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

    param_dict['regress_type'] = sys.argv[1]
    param_dict['scheme'] = sys.argv[2]
    param_dict['n_candidates'] = int(sys.argv[3])

    n_iter = 5

    for i in range(n_iter):
        tprint('Iteration {}'.format(i))

        param_dict = train(**param_dict)

        param_dict = iterate(**param_dict)
