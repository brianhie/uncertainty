from utils import *

from itertools import combinations_with_replacement as comb

from train_davis2011kinase import mlp_ensemble
from iterate_davis2011kinase import acquisition_rank

def load_data(**kwargs):
    gene2vec = {}
    with open('data/cho2016mashup/gene2vec.txt') as f:
        for line in f:
            fields = line.rstrip().split()
            vec = np.array([ float(x) for x in fields[1:] ])
            gene2vec[fields[0]] = vec

    fitness = []
    with open('data/norman2019gi/processed_emap.csv') as f:
        genes = f.readline().rstrip().split(',')[1:]
        for idx, line in enumerate(f):
            fields = line.rstrip().split(',')
            assert(fields[0] == genes[idx])
            fitness.append([ float(x) for x in fields[1:] ])
    fitness = np.array(fitness)

    gene2idx = { gene: idx for idx, gene in enumerate(genes) }

    unfeaturized = set(genes) - set(gene2vec.keys())
    tprint('No features found for:')
    tprint(', '.join(sorted(unfeaturized)))

    genes = np.array([ gene for gene in genes
                       if gene not in unfeaturized ])
    feat_idx = [ gene2idx[gene] for gene in genes ]
    fitness = fitness[feat_idx][:, feat_idx]

    kwargs['genes'] = genes
    kwargs['gene2vec'] = gene2vec
    kwargs['fitness'] = fitness

    return kwargs

def featurize(featurization='cat', **kwargs):
    genes = kwargs['genes']
    gene2vec = kwargs['gene2vec']
    fitness = kwargs['fitness']

    gene2idx = { gene: idx for idx, gene in enumerate(genes) }

    train_gene_idx = set(np.random.choice(
        len(genes), len(genes) // 2, replace=False
    ))

    X_obs, y_obs, X_unk, y_unk = [], [], [], []
    idx_obs, idx_unk = [], []
    idx_side, idx_novel = [], []
    for idx1, idx2 in comb(range(len(genes)), 2):
        gene1, gene2 = genes[idx1], genes[idx2]

        if featurization == 'cat':
            feature = np.concatenate([ gene2vec[gene1],
                                       gene2vec[gene2] ])
        elif featurization == 'mean':
            feature = (gene2vec[gene1] + gene2vec[gene2]) / 2
        elif featurization == 'diff':
            feature = np.concatenate([
                (gene2vec[gene1] + gene2vec[gene2]) / 2,
                np.abs(gene2vec[gene1] - gene2vec[gene2])
            ])
        else:
            raise ValueError('Invalid featurization: {}'
                             .format(featurization))

        fit = fitness[gene2idx[gene1], gene2idx[gene2]]

        if idx1 in train_gene_idx and idx2 in train_gene_idx:
            idx_obs.append((idx1, idx2))
            X_obs.append(feature)
            y_obs.append(fit)
            if idx1 != idx2:
                idx_obs.append((idx2, idx1))
                feature = np.concatenate([ gene2vec[gene2],
                                           gene2vec[gene1] ])
                X_obs.append(feature)
                y_obs.append(fit)

        elif idx1 in train_gene_idx or idx2 in train_gene_idx:
            idx_unk.append((idx1, idx2))
            idx_side.append((idx1, idx2))
            X_unk.append(feature)
            y_unk.append(fit)

        else:
            idx_unk.append((idx1, idx2))
            idx_novel.append((idx1, idx2))
            X_unk.append(feature)
            y_unk.append(fit)

    kwargs['X_obs'] = np.array(X_obs)
    kwargs['y_obs'] = np.array(y_obs)
    kwargs['X_unk'] = np.array(X_unk)
    kwargs['y_unk'] = np.array(y_unk)
    kwargs['idx_obs'] = idx_obs
    kwargs['idx_unk'] = idx_unk
    kwargs['idx_side'] = idx_side
    kwargs['idx_novel'] = idx_novel

    return kwargs

def train(regress_type, seed=1, **kwargs):
    X_obs = kwargs['X_obs']
    y_obs = kwargs['y_obs']

    kwargs['regress_type'] = regress_type

    # Debug.
    #X_obs = X_obs[:10]
    #y_obs = y_obs[:10]

    # Fit the model.

    if regress_type == 'mlper1':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=1,
            n_epochs=50,
            seed=seed,
            verbose=False,
        )
    elif regress_type == 'dmlper1':
        regressor = mlp_ensemble(
            n_neurons=1000,
            n_regressors=1,
            n_hidden_layers=15,
            n_epochs=300,
            seed=seed,
            verbose=False,
        )

    elif regress_type == 'mlper5g':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=5,
            n_epochs=50,
            loss='gaussian_nll',
            seed=seed,
            verbose=False,
        )
    elif regress_type == 'dmlper5g':
        regressor = mlp_ensemble(
            n_neurons=1000,
            n_regressors=5,
            n_hidden_layers=15,
            n_epochs=300,
            loss='gaussian_nll',
            seed=seed,
            verbose=False,
        )

    elif regress_type == 'bayesnn':
        from bayesian_neural_network import BayesianNN
        regressor = BayesianNN(
            n_hidden1=200,
            n_hidden2=200,
            n_iter=1000,
            n_posterior_samples=100,
            random_state=seed,
            verbose=True,
        )

    elif regress_type == 'cmf':
        from cmf_regressor import CMFRegressor
        regressor = CMFRegressor(
            n_components=100,
            seed=seed,
            verbose=False,
        )
        regressor.fit(
            kwargs['genes'],
            kwargs['genes'],
            kwargs['gene2vec'],
            kwargs['gene2vec'],
            kwargs['fitness'],
            kwargs['idx_obs'],
        )

    elif regress_type == 'gp':
        from gaussian_process import GPRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        regressor = GPRegressor(
            kernel=C(4., 'fixed') * RBF(1., 'fixed'),
            backend='sklearn',
            n_jobs=10,
            normalize_y=False,
            verbose=True
        )

    elif regress_type == 'hybrid':
        from gaussian_process import GPRegressor
        from hybrid import HybridMLPEnsembleGP
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=200,
                n_regressors=1,
                n_epochs=50,
                seed=seed,
            ),
            GPRegressor(
                backend='sklearn',#'gpytorch',
                n_restarts=10,
                n_jobs=10,
                verbose=True,
            ),
        )
    elif regress_type == 'dhybrid':
        from gaussian_process import GPRegressor
        from hybrid import HybridMLPEnsembleGP
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=1000,
                n_regressors=1,
                n_hidden_layers=15,
                n_epochs=300,
                seed=seed,
            ),
            GPRegressor(
                backend='sklearn',#'gpytorch',
                n_restarts=10,
                n_jobs=10,
                verbose=True,
            ),
        )

    if regress_type not in { 'cmf' }:
        regressor.fit(X_obs, y_obs)

    #print(regressor.model_.kernel_.get_params()) # Debug.

    kwargs['regressor'] = regressor

    return kwargs

def select_candidates(point=False, **kwargs):
    y_unk_pred = kwargs['y_unk_pred']
    var_unk_pred = kwargs['var_unk_pred']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    n_candidates = kwargs['n_candidates']
    genes = kwargs['genes']

    for fitness_type in [ 'positive', 'negative' ]:
        tprint('Selecting GI scores that are {}'.format(fitness_type))

        if fitness_type == 'negative':
            y_unk_pred *= -1

        if point:
            tprint('Exploiting (using point prediction only)...')
            acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta=0.)
        else:
            tprint('Exploiting...')
            acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta=1.)

        max_acqs = np.argsort(-acquisition)[:n_candidates]

        for max_acq in max_acqs:
            i, j = idx_unk[max_acq]
            gene1, gene2 = genes[i], genes[j]
            if y_unk is None:
                tprint('\tAcquire {} {} <--> {} with predicted GI score {:.3f}'
                       ' and variance {:.3f}'
                       .format((i, j), gene1, gene2, y_unk_pred[max_acq],
                               var_unk_pred[max_acq]))
            else:
                tprint('\tAcquire {} {} <--> {} with true GI score {}'
                       .format((i, j), gene1, gene2, y_unk[max_acq]))

    return list(max_acqs)

def select_candidates_per_quadrant(**kwargs):
    y_unk_pred = kwargs['y_unk_pred']
    var_unk_pred = kwargs['var_unk_pred']
    X_unk = kwargs['X_unk']
    y_unk = kwargs['y_unk']
    idx_unk = kwargs['idx_unk']
    n_candidates = kwargs['n_candidates']
    genes = kwargs['genes']

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred)

    acquired = []

    quad_names = [ 'side', 'novel' ]

    orig_idx = np.array(list(range(X_unk.shape[0])))

    for quad_name in quad_names:
        tprint('Considering quadrant {}'.format(quad_name))

        quad = [ i for i, idx in enumerate(idx_unk)
                 if idx in set(kwargs['idx_' + quad_name]) ]

        for fitness_type in [ 'positive', 'negative' ]:
            tprint('Selecting GI scores that are {}'.format(fitness_type))

            y_unk_quad = y_unk_pred[quad]
            if fitness_type == 'negative':
                y_unk_quad *= -1
            var_unk_quad = var_unk_pred[quad]
            idx_unk_quad = [ idx for i, idx in enumerate(idx_unk)
                             if idx in set(kwargs['idx_' + quad_name]) ]

            max_acqs = np.argsort(-acquisition[quad])[:n_candidates]

            for max_acq in max_acqs:
                i, j = idx_unk_quad[max_acq]
                gene1, gene2 = genes[i], genes[j]
                if y_unk is None:
                    tprint('\tAcquire {} {} <--> {} with predicted GI score {}'
                           .format((i, j), gene1, gene2, y_unk_quad[max_acq]))
                else:
                    tprint('\tAcquire {} {} <--> {} with real GI score {}'
                           .format((i, j), gene1, gene2, y_unk[quad][max_acq]))

            acquired += list(orig_idx[quad][max_acqs])

    return acquired

def acquire(**kwargs):
    if 'scheme' in kwargs:
        scheme = kwargs['scheme']
    else:
        scheme = 'exploit'
    if 'n_candidates' in kwargs:
        n_candidates = kwargs['n_candidates']
    else:
        kwargs['n_candidates'] = 1
    X_unk = kwargs['X_unk']
    idx_unk = kwargs['idx_unk']
    regress_type = kwargs['regress_type']
    regressor = kwargs['regressor']

    if regress_type == 'cmf':
        kwargs['y_unk_pred'] = regressor.predict(idx_unk)
    else:
        kwargs['y_unk_pred'] = regressor.predict(X_unk)
    kwargs['var_unk_pred'] = regressor.uncertainties_

    if scheme == 'exploit':
        acquired = select_candidates(**kwargs)

    elif scheme == 'pointexploit':
        acquired = select_candidates(point=True, **kwargs)

    elif scheme == 'quad':
        acquired = select_candidates_per_quadrant(**kwargs)

    return acquired, kwargs

if __name__ == '__main__':
    model = sys.argv[1]
    scheme = sys.argv[2]
    n_candidates = int(sys.argv[3])
    if len(sys.argv) >= 5:
        seed = int(sys.argv[4])
    else:
        seed = 1

    params = featurize(**load_data())

    params = train(model, seed, **params)

    acquire(scheme=scheme, n_candidates=n_candidates, **params)
