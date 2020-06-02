from utils import *

from anndata import AnnData
from sklearn.metrics import roc_auc_score
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

def auroc(y1, y2):
    labels = np.zeros(len(y1) + len(y2))
    labels[:len(y1)] = 1.
    values = np.concatenate([ y1, y2 ])
    return roc_auc_score(labels, values)

def chart_path(adata, source_idx, dest_idx,
               n_top_source=100, n_top_dest=100,
               print_report=False):
    from sklearn.metrics import roc_auc_score

    adata_source = adata[source_idx]
    adata_dest = adata[dest_idx]

    X_source = csc_matrix(adata_source.X)
    X_dest = csc_matrix(adata_dest.X)

    # Exclude genes that are directly perturbed.

    perturbs_both = set(adata_source.obs['perturb'])
    genes_perturbed = set([
        gene
        for perturb in perturbs_both
        for gene in perturb.split('_')
    ])
    if print_report:
        tprint('Perturbed {}'
               .format(', '.join(sorted(genes_perturbed))))

    # Score genes based on how they separate source and
    # destination cluster.

    source_ridx = np.random.choice(X_source.shape[0], 2000, False)
    dest_ridx = np.random.choice(X_dest.shape[0], 2000, False)

    fold_changes, p_values = [], []
    sig_fold_changes, sig_p_values = [], []
    scores_source, scores_dest = {}, {}
    for g_idx, gene in enumerate(adata.var_names):
        if not adata.var['highly_variable'][g_idx]:
            continue
        if gene in genes_perturbed:
            continue

        x_source = X_source[:, g_idx].toarray().flatten()
        x_dest = X_dest[:, g_idx].toarray().flatten()

        scores_source[gene] = x_source.mean() - x_dest.mean()
        scores_dest[gene] = x_dest.mean() - x_source.mean()

        fold_changes.append(scores_dest[gene])
        p_val = ss.ttest_ind(x_dest[dest_ridx], x_source[source_ridx])[1]
        if p_val == 0:
            p_val = 1e-308
        p_values.append(p_val)

        if scores_dest[gene] < -0.89 or \
           scores_dest[gene] > 0.54:
            sig_fold_changes.append(scores_dest[gene])
            sig_p_values.append(p_val)

    # Sort and pick top genes.

    top_source = sorted(scores_source.items(),
                        key=lambda x: -x[1])[:n_top_source]
    if print_report:
        tprint('Source markers:')
        for gene, score in top_source:
            tprint('{}:\t{}'.format(gene, score))
        tprint('')

    top_dest = sorted(scores_dest.items(),
                      key=lambda x: -x[1])[:n_top_dest]
    if print_report:
        tprint('Destination markers:')
        for gene, score in top_dest:
            tprint('{}:\t{}'.format(gene, score))
        tprint('')

    plt.figure()
    plt.scatter(fold_changes, -np.log10(p_values),
                c='#111111', alpha=0.1)
    plt.scatter(sig_fold_changes, -np.log10(sig_p_values),
                c='#bd2031', alpha=1.)
    plt.savefig('figures/norman_de.png', dpi=300)
    plt.close()

    diffex_genes = np.array(
        [ x[0] for x in top_source ] + [ x[0] for x in top_dest ]
    )
    high_low = np.array(
        [ 'low' ] * len(top_source) + [ 'high' ] * len(top_dest)
    )

    return diffex_genes, high_low


def visualize_diffex_genes(adata, source_idx, dest_idx,
                           diffex_genes, high_low, namespace):
    X_source = csc_matrix(adata[source_idx].X)
    X_dest = csc_matrix(adata[dest_idx].X)

    for hl, gene in zip(high_low, diffex_genes):
        g_idx = list(adata.var_names).index(gene)
        x_source = X_source[:, g_idx].toarray().flatten()
        x_dest = X_dest[:, g_idx].toarray().flatten()

        plt.figure()
        sns.violinplot(data=[ x_source, x_dest ],
                       scale='width', cut=0)
        sns.stripplot(data=[ x_source, x_dest ],
                      jitter=True, color='black', size=1)
        plt.xticks([0, 1], [ 'Source', 'Destination' ])
        plt.savefig('figures/{}_diffex_{}_{}.png'
                    .format(namespace, hl, gene), dpi=300)
        plt.close()


def hl_dist(a, b, high_low):
    # Directionality of distance matters!
    low_idx, high_idx = high_low == 'low', high_low == 'high'
    return (np.sum(a[low_idx] - b[low_idx]) +
            np.sum(b[high_idx] - a[high_idx]))

def find_optimal(adata, diffex_genes, source_idx, high_low,
                 verbose=False):
    gene_idxs = [ list(adata.var_names).index(gene)
                  for gene in diffex_genes ]
    X_genes = csr_matrix(csc_matrix(adata.X)[:, gene_idxs])

    source_vec = np.array(X_genes[source_idx].mean(0)).flatten()

    perturb_dist = []
    uniq_perturb = np.array(sorted(set(adata.obs['perturb'])))
    for perturb in uniq_perturb:
        perturb_idx = np.array(adata.obs['perturb'] == perturb)
        perturb_vec = np.array(X_genes[perturb_idx].mean(0)).flatten()

        dist = hl_dist(source_vec, perturb_vec, high_low)
        perturb_dist.append(dist)

    rank_perturb = uniq_perturb[np.argsort(-np.array(perturb_dist))]

    if verbose:
        tprint('Best perturbs (ranked): {}'
               .format(', '.join(rank_perturb[:30])))

    return rank_perturb[0]

def mlp_ensemble(n_neurons=500, n_regressors=5, n_epochs=100,
                 n_hidden_layers=2, loss='mse', seed=1, verbose=False,):
    from mlp_ensemble_perturb import MLPEnsembleRegressor

    layer_sizes_list = []
    for i in range(n_regressors):
        layer_sizes_list.append((n_neurons,) * n_hidden_layers)

    mlper = MLPEnsembleRegressor(
        layer_sizes_list,
        activations='relu',
        loss=loss,
        solvers='adam',
        alphas=0.1,
        batch_sizes=500,
        max_iters=n_epochs,
        momentums=0.9,
        nesterovs_momentums=True,
        backend='keras',
        random_state=seed,
        verbose=verbose,
    )

    return mlper


def get_regressor(regress_method, seed=1):
    if regress_method == 'gp':
        from gaussian_process_perturb import GPRegressor, SparseGPRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        regressor = SparseGPRegressor(
            method='geosketch',
            n_inducing=20000,
            backend='gpytorch',
            n_restarts=10,
            n_jobs=1,
            seed=seed,
            verbose=2
        )

    elif regress_method == 'sparsegp':
        from gaussian_process_perturb import SparseGPRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        regressor = SparseGPRegressor(
            method='geosketch',
            n_inducing=5000,
            backend='gpytorch',
            n_restarts=10,
            seed=seed,
            n_jobs=1,
            verbose=2
        )

    elif regress_method == 'hybrid':
        from gaussian_process_perturb import GPRegressor
        from hybrid import HybridMLPEnsembleGP
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=1000,
                n_hidden_layers=15,
                n_regressors=1,
                n_epochs=400,
                seed=seed,
                verbose=False,
            ),
            GPRegressor(
                backend='gpytorch',
                n_restarts=10,
                n_jobs=1,
                seed=seed,
                verbose=True,
            ),
        )

    elif regress_method == 'sparsehybrid':
        from gaussian_process_perturb import SparseGPRegressor
        from hybrid import HybridMLPEnsembleGP
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=1000,
                n_hidden_layers=15,
                n_regressors=1,
                n_epochs=400,
                seed=seed,
                verbose=False,
            ),
            SparseGPRegressor(
                method='geosketch',
                n_inducing=5000,
                backend='gpytorch',#'sklearn',
                n_restarts=10,
                n_jobs=1,
                seed=seed,
                verbose=True
            ),
        )

    elif regress_method == 'mlper1':
        regressor = mlp_ensemble(
            n_neurons=1000,
            n_regressors=1,
            n_hidden_layers=15,
            n_epochs=400,
            seed=seed,
            verbose=False,
        )

    elif regress_method == 'mlper5g':
        regressor = mlp_ensemble(
            n_neurons=1000,
            n_regressors=5,
            n_hidden_layers=15,
            n_epochs=50,
            loss='gaussian_nll',
            seed=seed,
        )

    elif regress_method == 'bayesnn':
        from bayesian_neural_network_perturb import BayesianNN
        regressor = BayesianNN(
            n_hidden1=200,
            n_hidden2=200,
            n_iter=100,
            n_posterior_samples=100,
            sketch_size=5000,
            random_state=seed,
            verbose=True,
        )

    elif regress_method == 'linear':
        from linear_regression import LinearRegressor
        regressor = LinearRegressor()

    else:
        raise ValueError('Acquisition method {} not available'
                         .format(regress_method))

    return regressor

def get_perturb_idxs(genes, perturbs):
    gene2idx = { gene: idx for idx, gene in enumerate(genes) }
    perturb_idxs = []
    for perturb, perturb_type in perturbs:
        perturb_idxs.append(([
            gene2idx[gene] for gene in perturb.split('_')
            if gene in gene2idx and 'NegCtrl' not in gene
        ], perturb_type))
    return perturb_idxs

def epitome(X, perturb_idx, perturb_type):
    if perturb_type != 'crispra':
        raise ValueError(
            'Epitome strategy only valid for activation perturbations'
        )
    rank_sum = np.zeros(X.shape[0])
    for p_idx in perturb_idx:
        rank_sum += ss.rankdata(-np.ravel(X[:, p_idx]))
    return X[np.argsort(rank_sum)[:500]].mean(0)

def compute_transition(X, method):
    if method == 'spearman':
        trans = ss.spearmanr(X)[0]
        trans[np.isnan(trans)] = 0
    elif method == 'pearson':
        trans = ss.pearsonr(X)[0]
        trans[np.isnan(trans)] = 0
    else:
        raise ValueError('Invalid transition method {}'
                         .format(method))
    #trans = np.abs(trans)
    trans[np.abs(trans) < 0.3] = 0
    trans = normalize(trans, norm='l1')
    #trans /= np.min(trans)
    return trans

def rwr(x, trans, prob):
    A = np.eye(len(x)) - ((1 - prob) * trans)
    b = prob * x
    return np.linalg.solve(A, b)

def perturb_crispra(x, perturb_idx, gain=5.):
    x = np.array(x[:]).flatten()
    if len(perturb_idx) > 0:
        x[perturb_idx] = 1.#*= gain
    return x

def perturb_crispri(x, perturb_idx, expr=0.):
    x = np.array(x[:]).flatten()
    if len(perturb_idx) > 0:
        x[perturb_idx] = expr
    return x

def check_none(x, name):
    if x is None:
        raise ValueError('Must supply {}'.format(name))

def featurize(
        X, genes, perturbs, mode='perfect',
        rwr_transition=None, rwr_prob=None,
):
    if len(genes) != X.shape[1]:
        raise ValueError('Gene dimension disagreement')
    if mode != 'nn' and len(perturbs) != X.shape[0]:
        raise ValueError('Perturbation dimension disagreement')
    if 'rwr' in mode:
        check_none(rwr_transition, 'RWR transition')
        check_none(rwr_prob, 'RWR probability')

    perturb_idxs = get_perturb_idxs(genes, perturbs)

    X_transform = []
    cache = {}

    for i in range(len(perturbs)):

        if mode == 'perfect':
            x_transform = X[i]

        elif mode == 'nn':
            if perturb_idxs is None or len(perturb_idxs[i][0]) == 0:
                x_transform = X.mean(0)
            else:
                perturb_idx, perturb_type = perturb_idxs[i]
                x_transform = epitome(X, perturb_idx, perturb_type)

        elif mode.startswith('art'):
            if perturb_idxs is None:
                perturb_idx = []
                perturb_type = 'crispra'
            else:
                perturb_idx, perturb_type = perturb_idxs[i]

            if perturb_type == 'crispra':
                perturb_fn = perturb_crispra
            elif perturb_type == 'crispri':
                perturb_fn = perturb_crispri
            else:
                raise ValueError('Unknown perturbation type {}'
                                 .format(perturb_type))

            if 'scale' in mode:
                x_base = X[i]

            if 'rwr' in mode:
                # Cache RWR computation for efficiency.
                if tuple(perturb_idx) in cache:
                    x_scale = cache[tuple(perturb_idx)]
                else:
                    x_scale = perturb_fn(np.zeros(X.shape[1]), perturb_idx)
                    x_scale = rwr(x_scale, rwr_transition, rwr_prob)
                    cache[tuple(perturb_idx)] = x_scale

            if mode == 'art-scale-expr':
                x_transform = perturb_fn(x_base, perturb_idx)

            elif mode == 'art-scale-rwr':
                x_transform = x_scale

            elif mode == 'art-scale-rwr-expr':
                x_transform = np.multiply(x_base, x_scale)

            else:
                raise ValueError('Invalid mode {}'.format(mode))
        else:
            raise ValueError('Invalid mode {}'.format(mode))

        X_transform.append(x_transform)

    return np.vstack(X_transform)

def acquisition_rank(y_pred, var_pred, beta=1.):
    return ss.rankdata(y_pred) + (beta * ss.rankdata(-var_pred))

def acquisition_ucb(y_pred, var_pred, beta=1.):
    return y_pred - (beta * var_pred)

def acquisition_fn(scores, var, acq_fn_name, beta):
    if acq_fn_name == 'rank-ucb':
        return acquisition_rank(scores, var, beta)
    elif acq_fn_name == 'ucb':
        return acquisition_ucb(scores, var, beta)
    else:
        raise ValueError('Invalid acquistion function name {}'
                         .format(acq_fn_name))

def acquire_perturbations(
        adata,
        X_train, y_train,
        X_test,
        y_base, high_low,
        mode='perfect',
        n_acquire=20,
        regress_method='gp',
        acq_fn_name='rank-ucb',
        beta=1.,
        seed=1,
):
    # Fit regressor.

    regressor = get_regressor(regress_method, seed=seed)
    regressor.fit(X_train, y_train)

    # Predict effect of perturbations.

    y_pred = regressor.predict(X_test)
    var = regressor.uncertainties_

    # Rank perturbations.

    scores = np.array([
        hl_dist(y_base, y_pred[i].flatten(), high_low)
        for i in range(y_pred.shape[0])
    ])
    acquisition = acquisition_fn(scores, var, acq_fn_name, beta)
    perturb_rank = np.argsort(-acquisition)

    # Return best perturbations.

    acq_perturbs = []
    for perturb in perturb_rank:
        if perturb not in acq_perturbs:
            acq_perturbs.append(perturb)
        if len(acq_perturbs) >= n_acquire:
            break

    return acq_perturbs, acquisition, var
