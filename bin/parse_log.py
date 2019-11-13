import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import sys

from utils import plt
from process_davis2011kinase import process

def parse_log(regress_type, experiment, **kwargs):
    log_fname = ('iterate_davis2011kinase_{}_{}.log'
                 .format(regress_type, experiment))

    iteration = 0
    iter_to_Kds = {}
    iter_to_idxs = {}

    with open(log_fname) as f:

        while True:
            line = f.readline()
            if not line:
                break

            if not line.startswith('2019') and not line.startswith('2020'):
                continue
            if not ' | ' in line:
                continue

            line = line.split(' | ')[1]

            if line.startswith('Iteration'):
                iteration = int(line.strip().split()[-1])
                if not iteration in iter_to_Kds:
                    iter_to_Kds[iteration] = []
                if not iteration in iter_to_idxs:
                    iter_to_idxs[iteration] = []

                continue

            elif line.startswith('\tAcquire '):
                fields = line.strip().split()

                Kd = float(fields[-1])
                iter_to_Kds[iteration].append(Kd)

                chem_idx = int(fields[1].lstrip('(').rstrip(','))
                prot_idx = int(fields[2].strip().rstrip(')'))
                iter_to_idxs[iteration].append((chem_idx, prot_idx))

                continue

    assert(iter_to_Kds.keys() == iter_to_idxs.keys())
    iterations = sorted(iter_to_Kds.keys())

    # Plot Kd over iterations.

    Kd_iter, Kd_iter_max, Kd_iter_min = [], [], []
    all_Kds = []
    for iteration in iterations:
        Kd_iter.append(np.mean(iter_to_Kds[iteration]))
        Kd_iter_max.append(max(iter_to_Kds[iteration]))
        Kd_iter_min.append(min(iter_to_Kds[iteration]))
        all_Kds += list(iter_to_Kds[iteration])

        if iteration == 0:
            print('First average Kd is {}'.format(Kd_iter[0]))
        elif iteration > 4 and experiment == 'perprot':
            break

    print('Average Kd is {}'.format(np.mean(all_Kds)))

    plt.figure()
    plt.scatter(iterations, Kd_iter)
    plt.plot(iterations, Kd_iter)
    plt.fill_between(iterations, Kd_iter_min, Kd_iter_max, alpha=0.3)
    plt.viridis()
    plt.title(' '.join([ regress_type, experiment ]))
    plt.savefig('figures/Kd_over_iterations_{}_{}.png'
                .format(regress_type, experiment))
    plt.close()

    return

    # Plot differential entropy of acquired samples over iterations.

    chems = kwargs['chems']
    prots = kwargs['prots']
    chem2feature = kwargs['chem2feature']
    prot2feature = kwargs['prot2feature']

    d_entropies = []
    X_acquired = []
    for iteration in iterations:
        for i, j in iter_to_idxs[iteration]:
            chem = chems[i]
            prot = prots[j]
            X_acquired.append(chem2feature[chem] + prot2feature[prot])
        if len(X_acquired) <= 1:
            d_entropies.append(float('nan'))
        else:
            gaussian = GaussianMixture().fit(np.array(X_acquired))
            gaussian = multivariate_normal(
                gaussian.means_[0], gaussian.covariances_[0]
            )
            d_entropies.append(gaussian.entropy())

    print('Final differential entropy is {}'.format(d_entropies[-1]))

    plt.figure()
    plt.scatter(iterations, d_entropies)
    plt.plot(iterations, d_entropies)
    plt.viridis()
    plt.title(' '.join([ regress_type, experiment ]))
    plt.savefig('figures/entropy_over_iterations_{}_{}.png'
                .format(regress_type, experiment))
    plt.close()

if __name__ == '__main__':
    regress_type = sys.argv[1]
    experiment = sys.argv[2]

    parse_log(regress_type, experiment, **process())
