from utils import *

from train_davis2011kinase import mlp_ensemble

def load_embeddings(fname):
    X, meta = [], []
    with open(fname) as f:
        for line in f:
            if line.startswith('>'):
                mutations, n_mut, brightness = line[1:].split('_')
                n_mut = int(n_mut)
                brightness = float(brightness)
                X.append([ float(x) for x in f.readline().split() ])
                meta.append([ mutations, n_mut, brightness ])
    X = np.array(X)
    meta = pd.DataFrame(meta, columns=[
        'mutations', 'n_mut', 'brightness',
    ])
    return X, meta

def plot_stats(meta):
    plt.figure()
    sns.distplot(np.array(meta.n_mut).ravel(), kde=False)
    plt.savefig('figures/gfp_nmut_hist.svg')
    plt.close()

def plot_stats_fpbase(meta):
    edit_dist = np.array(meta.edit).ravel()

    print('Median edit dist {}'.format(np.median(edit_dist)))
    print('Mean edit dist {}'.format(np.mean(edit_dist)))

    plt.figure()
    sns.distplot(edit_dist, kde=False,
                 bins=10)
    plt.savefig('figures/gfp_edit_hist.svg')
    plt.close()

def split_X(X, meta):
    X_train, X_test, y_train, y_test = [], [], [], []
    mutations_test = []
    for i in range(X.shape[0]):
        n_mut = meta.n_mut[i]
        if n_mut > 2:
            X_test.append(X[i])
            y_test.append(meta.brightness[i])
            mutations_test.append(meta.mutations[i])
        else:
            X_train.append(X[i])
            y_train.append(meta.brightness[i])
    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test),
            mutations_test)

def train(regress_type, X_train, y_train, seed=1):
    if regress_type == 'mlper1':
        regressor = mlp_ensemble(
            n_neurons=200,
            n_regressors=1,
            n_epochs=50,
            seed=seed,
        )
    elif regress_type == 'dmlper1':
        regressor = mlp_ensemble(
            n_neurons=1000,
            n_regressors=1,
            n_hidden_layers=15,
            n_epochs=50,
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
        )
    elif regress_type == 'dmlper5g':
        regressor = mlp_ensemble(
            n_neurons=1000,
            n_regressors=15,
            n_epochs=50,
            loss='gaussian_nll',
            seed=seed,
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
    elif regress_type == 'lbayesnn':
        from bayesian_neural_network import BayesianNN
        regressor = BayesianNN(
            n_hidden1=1000,
            n_hidden2=1000,
            n_iter=1000,
            n_posterior_samples=100,
            random_state=seed,
            verbose=True,
        )

    elif regress_type == 'gp':
        from gaussian_process import GPRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        regressor = GPRegressor(
            kernel=C(2., 'fixed') * RBF(1., 'fixed'),
            normalize_y=False,
            backend='sklearn',
            n_jobs=10,
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
                normalize_y=False,
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
                n_epochs=600,
                seed=seed,
            ),
            GPRegressor(
                backend='sklearn',
                normalize_y=False,
                n_restarts=10,
                n_jobs=10,
                verbose=True,
            ),
        )
    elif regress_type == 'sparsehybrid':
        from gaussian_process import SparseGPRegressor
        from hybrid import HybridMLPEnsembleGP
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        regressor = HybridMLPEnsembleGP(
            mlp_ensemble(
                n_neurons=200,
                n_regressors=1,
                n_epochs=50,
                seed=seed,
            ),
            SparseGPRegressor(
                method='geosketch',
                n_inducing=8000,
                backend='sklearn',
                n_restarts=10,
                n_jobs=10,
                verbose=True
            ),
        )

    elif regress_type == 'linear':
        from linear_regression import LinearRegressor
        regressor = LinearRegressor()

    regressor.fit(X_train, y_train)

    return regressor

def acquisition_rank(y_pred, var_pred, beta=1.):
    return ss.rankdata(y_pred) + (beta * ss.rankdata(-var_pred))

def gfp_cv(model, beta, seed):
    X, meta = load_embeddings(
        'data/sarkisyan2016gfp/embeddings.txt'
    )

    plot_stats(meta)

    X_train, y_train, X_test, y_test, mutations_test = split_X(
        X, meta
    )

    # Use brightness cutoff of 3 used in original study.
    y_train -= 3.
    y_test -= 3.
    y_train[y_train < 0.] = 0.
    y_test[y_test < 0.] = 0.

    regressor = train(model, X_train, y_train, seed=seed)
    y_unk_pred = regressor.predict(X_test)
    var_unk_pred = regressor.uncertainties_

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta)

    acq_argsort = np.argsort(-acquisition)
    for rank, idx in enumerate(acq_argsort):
        fields = [
            rank, mutations_test[idx], y_test[idx]
        ]
        print('\t'.join([ str(field) for field in fields ]))

def load_fpbase(fname):
    X, meta = [], []
    with open(fname) as f:
        for line in f:
            if line.startswith('>'):
                name, ex, em, brightness, avgfp_edit = line[1:].rstrip().split('_')
                try:
                    ex = float(ex)
                    em = float(em)
                except:
                    continue
                if brightness.strip() == '':
                    continue
                    #brightness = -1
                elif em < 500 or em > 520:
                    brightness = 0
                else:
                    brightness = float(brightness)
                avgfp_edit = int(avgfp_edit)
                X.append([ float(x) for x in f.readline().split() ])
                meta.append([ name, ex, em, brightness, avgfp_edit ])
    X = np.array(X)
    meta = pd.DataFrame(meta, columns=[
        'name', 'ex', 'em', 'brightness', 'edit',
    ])
    return X, meta

def gfp_fpbase(model, beta, seed):
    X, meta = load_embeddings(
        'data/sarkisyan2016gfp/embeddings.txt'
    )

    X_train, y_train, X_test, y_test, mutations_test = split_X(
        X, meta
    )

    X_val, meta_val = load_fpbase(
        'data/sarkisyan2016gfp/fpbase_embeddings.txt'
    )

    plot_stats_fpbase(meta_val)

    regressor = train(model, X_train, y_train, seed=seed)
    y_unk_pred = regressor.predict(X_val)
    var_unk_pred = regressor.uncertainties_

    plt.figure()
    plt.scatter(meta_val.edit, var_unk_pred, alpha=0.3)
    plt.xlabel('Edit distance')
    plt.ylabel('Uncertainty')
    plt.savefig('figures/gfp_edit_uncertainty.png', dpi=300)
    print('Edit distance vs. uncertainty, '
          'Spearman r = {}'.format(
        ss.spearmanr(meta_val.edit, var_unk_pred)
    ))
    print('Edit distance vs. uncertainty, '
          'Pearson rho = {}'.format(
        ss.pearsonr(meta_val.edit, var_unk_pred)
    ))

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta)

    acq_argsort = np.argsort(-acquisition)
    for rank, idx in enumerate(acq_argsort):
        if meta_val.brightness[idx] < 0:
            continue
        fields = [
            rank, meta_val.name[idx], meta_val.brightness[idx]
        ]
        print('\t'.join([ str(field) for field in fields ]))

def egfp(model, beta, seed):
    X, meta = load_embeddings(
        'data/sarkisyan2016gfp/embeddings.txt'
    )

    X_train, y_train, _, _, _ = split_X(
        X, meta
    )

    X_mut3 = X[meta.n_mut == 3]

    X_fpbase, meta_fpbase = load_fpbase(
        'data/sarkisyan2016gfp/fpbase_egfp_embeddings.txt'
    )
    egfp_idx = list(meta_fpbase.name).index('EGFP')
    X_egfp = X_fpbase[egfp_idx].reshape(1, -1)

    X_val = np.concatenate([ X_egfp, X_mut3 ])

    regressor = train(model, X_train, y_train, seed=seed)
    y_unk_pred = regressor.predict(X_val)
    var_unk_pred = regressor.uncertainties_

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta)

    acq_ranks = ss.rankdata(acquisition)

    plt.figure()
    plt.scatter(acq_ranks, np.zeros(len(acq_ranks)),
                c='#dddddd', marker='s', alpha=0.1)
    plt.scatter([ acq_ranks[0] ], [ 0 ],
                c='red', marker='s', alpha=1.)
    plt.title('{}, EGFP rank {}'.format(model, acq_ranks[0]))
    plt.xlim([ -10, 100 ])
    plt.savefig('figures/egfp_loc_{}.png'.format(model), dpi=300)
    plt.close()

if __name__ == '__main__':
    model = sys.argv[1]
    beta = float(sys.argv[2])

    if len(sys.argv) >= 4:
        seed = int(sys.argv[3])
        np.random.seed(seed)
    else:
        seed = 1

    #gfp_cv(model, beta, seed)

    gfp_fpbase(model, beta, seed)

    #egfp(model, beta, seed)
