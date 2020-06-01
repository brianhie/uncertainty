from utils import mkdir_p, plt, tprint

import matplotlib.cm as cm
import numpy as np
import os
import scipy.stats as ss
import seaborn as sns
import sys

from iterate_davis2011kinase import acquire, acquisition_rank, acquisition_scatter
from process_davis2011kinase import process, visualize_heatmap
from train_davis2011kinase import train

def load_acquired(acquired_fname):
    acquired = []
    with open(acquired_fname) as f:
        for line in f:
            chem, prot, Kd = line.rstrip().split()
            Kd = float(Kd)
            acquired.append((chem, prot, Kd))
    return acquired

def load_chem_zinc(fname, chems):
    chem2zinc = {}
    with open(fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().rstrip(',').split(',')
            name = fields[0]
            zinc = fields[-2]
            chem2zinc[name] = zinc
    assert(len(set(chems) - set(chem2zinc.keys())) == 0)
    return chem2zinc

def load_zinc_features(fname, exclude=set()):
    zincs = []
    zinc2feature = {}
    with open(fname) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].rstrip()
                if name in exclude:
                    continue
                zincs.append(name)
                zinc2feature[name] = [
                    float(field) for field in f.readline().rstrip().split()
                ]
    return zincs, zinc2feature

def setup(**kwargs):
    Kds = kwargs['Kds']
    prots = kwargs['prots']
    chems = kwargs['chems']
    prot2feature = kwargs['prot2feature']
    chem2feature = kwargs['chem2feature']
    regress_type = kwargs['regress_type']
    prot_target = kwargs['prot_target']
    acquired = kwargs['acquired']

    chem2zinc = load_chem_zinc(
        'data/davis2011kinase/chem_smiles.csv', chems
    )

    zincs, zinc2feature = load_zinc_features(
        #'data/docking/mol_samples_jtnnvae_molonly.txt',
        'data/davis2011kinase/cayman_jtnnvae_molonly.txt',
        set({ chem2zinc[chem] for chem in chem2zinc })
    )

    orig_len_chems = len(chems)
    chems += zincs
    chem2feature.update(zinc2feature)

    # For runtime debugging.
    #idx_obs = [
    #    (i, j) for i in range(10) for j in range(10)
    #]
    #idx_unk = [
    #    (i + orig_len_chems, j) for i in range(10) for j in range(10)
    #]

    idx_obs = [
        (i, j) for i in range(orig_len_chems) for j in range(len(prots))
    ]

    prot2idx = { prot: prot_idx for prot_idx, prot in enumerate(prots) }
    chem2idx = { chem: chem_idx for chem_idx, chem in enumerate(chems) }
    acquired_pairs = {}
    for chem, prot, Kd in acquired:
        idx_obs.append((chem2idx[chem], prot2idx[prot]))
        acquired_pairs[(chem, prot)] = Kd

    idx_unk = [
        (i + orig_len_chems, j) for i in range(len(zincs))
        for j in range(len(prots))
        if (chems[i + orig_len_chems], prots[j]) not in acquired_pairs and
        (prot_target is None or prots[j] == prot_target)
    ]

    tprint('Constructing training dataset...')
    X_obs, y_obs = [], []
    for i, j in idx_obs:
        chem = chems[i]
        prot = prots[j]
        X_obs.append(chem2feature[chem] + prot2feature[prot])
        if (chem, prot) in acquired_pairs:
            y_obs.append(acquired_pairs[(chem, prot)])
        else:
            y_obs.append(Kds[i, j])
    X_obs, y_obs = np.array(X_obs), np.array(y_obs)

    tprint('Constructing evaluation dataset...')
    X_unk = []
    for i, j in idx_unk:
        chem = chems[i]
        prot = prots[j]
        X_unk.append(chem2feature[chem] + prot2feature[prot])
    X_unk = np.array(X_unk)

    kwargs['X_obs'] = X_obs
    kwargs['y_obs'] = y_obs
    kwargs['idx_obs'] = idx_obs
    kwargs['X_unk'] = X_unk
    kwargs['y_unk'] = None
    kwargs['idx_unk'] = idx_unk
    kwargs['chems'] = chems
    kwargs['chem2feature'] = chem2feature

    return kwargs

def latent_scatter(var_unk_pred, y_unk_pred, acquisition, **kwargs):
    chems = kwargs['chems']
    chem2feature = kwargs['chem2feature']
    idx_obs = kwargs['idx_obs']
    idx_unk = kwargs['idx_unk']
    regress_type = kwargs['regress_type']
    prot_target = kwargs['prot_target']

    chem_idx_obs = sorted(set([ i for i, _ in idx_obs ]))
    chem_idx_unk = sorted(set([ i for i, _ in idx_unk ]))

    feature_obs = np.array([
        chem2feature[chems[i]] for i in chem_idx_obs
    ])
    feature_unk = np.array([
        chem2feature[chems[i]] for i in chem_idx_unk
    ])

    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(feature_obs)
    dist = np.ravel(nbrs.kneighbors(feature_unk)[0])
    print('Distance Spearman r = {}, P = {}'
          .format(*ss.spearmanr(dist, var_unk_pred)))
    print('Distance Pearson rho = {}, P = {}'
          .format(*ss.pearsonr(dist, var_unk_pred)))
    exit()

    X = np.vstack([ feature_obs, feature_unk ])
    labels = np.concatenate([
        np.zeros(len(chem_idx_obs)), np.ones(len(chem_idx_unk))
    ])
    sidx = np.argsort(-var_unk_pred)

    from fbpca import pca
    U, s, Vt = pca(X, k=3,)
    X_pca = U * s

    from umap import UMAP
    um = UMAP(
        n_neighbors=15,
        min_dist=0.5,
        n_components=2,
        metric='euclidean',
    )
    X_umap = um.fit_transform(X)

    from MulticoreTSNE import MulticoreTSNE as TSNE
    tsne = TSNE(
        n_components=2,
        n_jobs=20,
    )
    X_tsne = tsne.fit_transform(X)

    if prot_target is None:
        suffix = ''
    else:
        suffix = '_' + prot_target

    for name, coords in zip(
            [ 'pca', 'umap', 'tsne' ],
            [ X_pca, X_umap, X_tsne ],
    ):
        plt.figure()
        sns.scatterplot(x=coords[labels == 1, 0], y=coords[labels == 1, 1],
                        color='blue', alpha=0.1,)
        plt.scatter(x=coords[labels == 0, 0], y=coords[labels == 0, 1],
                    color='orange', alpha=1.0, marker='x', linewidths=10,)
        plt.savefig('figures/latent_scatter_{}_ypred_{}{}.png'
                    .format(name, regress_type, suffix), dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(x=coords[labels == 1, 0], y=coords[labels == 1, 1],
                    c=ss.rankdata(var_unk_pred), alpha=0.1, cmap='coolwarm')
        plt.savefig('figures/latent_scatter_{}_var_{}{}.png'
                    .format(name, regress_type, suffix), dpi=300)
        plt.close()

        plt.figure()
        plt.scatter(x=coords[labels == 1, 0], y=coords[labels == 1, 1],
                    c=-acquisition, alpha=0.1, cmap='hot')
        plt.savefig('figures/latent_scatter_{}_acq_{}{}.png'
                    .format(name, regress_type, suffix), dpi=300)
        plt.close()

def predict(**kwargs):
    X_unk = kwargs['X_unk']
    regress_type = kwargs['regress_type']
    prot_target = kwargs['prot_target']

    mkdir_p('target/prediction_cache')

    if prot_target is None:
        suffix = ''
    else:
        suffix = '_{}'.format(prot_target)

    if os.path.isfile('target/prediction_cache/{}_ypred{}.npy'
                      .format(regress_type, suffix)):
        y_unk_pred = np.load('target/prediction_cache/{}_ypred{}.npy'
                             .format(regress_type, suffix))
        var_unk_pred = np.load('target/prediction_cache/{}_varpred{}.npy'
                               .format(regress_type, suffix))
    else:
        y_unk_pred = None

    if y_unk_pred is None or y_unk_pred.shape[0] != X_unk.shape[0]:
        kwargs = train(**kwargs)
        regressor = kwargs['regressor']

        if regress_type == 'cmf':
            y_unk_pred = regressor.predict(kwargs['idx_unk'])
        else:
            y_unk_pred = regressor.predict(X_unk)
        var_unk_pred = regressor.uncertainties_
        np.save('target/prediction_cache/{}_ypred{}.npy'
                .format(regress_type, suffix), y_unk_pred)
        np.save('target/prediction_cache/{}_varpred{}.npy'
                .format(regress_type, suffix), var_unk_pred)

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred)
    acquisition_scatter(y_unk_pred, var_unk_pred, acquisition,
                        regress_type)

    # Plot visualization of chemical latent space.
    latent_scatter(var_unk_pred, y_unk_pred, acquisition, **kwargs)

    kwargs['y_unk_pred'] = y_unk_pred
    kwargs['var_unk_pred'] = var_unk_pred

    return kwargs

def repurpose(**kwargs):
    idx_unk = kwargs['idx_unk']
    chems = kwargs['chems']
    prots = kwargs['prots']

    kwargs = predict(**kwargs)

    to_acquire = acquire(**kwargs)[0]

    for idx in to_acquire:
        i, j = idx_unk[idx]
        tprint('Please acquire {} <--> {}'.format(chems[i], prots[j]))

if __name__ == '__main__':
    param_dict = process()

    param_dict['regress_type'] = sys.argv[1]
    param_dict['scheme'] = sys.argv[2]
    param_dict['n_candidates'] = int(sys.argv[3])

    if len(sys.argv) >= 5:
        param_dict['prot_target'] = sys.argv[4].upper()
        if param_dict['prot_target'] == 'PKNB':
            param_dict['prot_target'] = 'PKNB(M.tuberculosis)'
    else:
        sys.stderr.write('Warning: Protein target not set,'
                         'considering all kinases...\n')
        param_dict['prot_target'] = None

    if len(sys.argv) >= 6:
        acquired_fname = sys.argv[5]
        param_dict['acquired'] = load_acquired(acquired_fname)
    else:
        param_dict['acquired'] = []

    param_dict = setup(**param_dict)

    repurpose(**param_dict)
