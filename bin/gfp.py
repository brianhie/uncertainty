from utils import *

from train_davis2011kinase import mlp_ensemble

def plot_stats(meta):
    plt.figure()
    sns.distplot(np.array(meta.n_mut).ravel(), kde=False)
    plt.savefig('figures/gfp_nmut_hist.svg')
    plt.close()

    plt.figure()
    sns.violinplot(meta.brightness, kde=False)
    plt.savefig('figures/gfp_fluorescence_violin.svg')
    plt.close()

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
    plot_stats(meta)
    return X, meta

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
            n_iter=100,
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

    #if model == 'gp':
    np.save('target/prediction_cache/gfp_cv_ypred_{}.npy'
            .format(model), y_unk_pred)
    np.save('target/prediction_cache/gfp_cv_varpred_{}.npy'
            .format(model), var_unk_pred)
    np.save('target/prediction_cache/gfp_cv_acq_{}.npy'
            .format(model), acquisition)

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

def gfp_structure(model, beta, seed):
    from Bio import Seq, SeqIO
    gfp_fasta = 'data/sarkisyan2016gfp/avGFP_reference_sequence.fa'
    for record in SeqIO.parse(gfp_fasta, 'fasta'):
        ref_nt_seq = record.seq
        break
    ref_seq = Seq.translate(ref_nt_seq)

    with open('data/sarkisyan2016gfp/2wur.pdb') as f:
        surface_idxs = set([ int(line[23:26]) for line in f
                             if line.startswith('ANISOU') ])

    X, meta = load_embeddings(
        'data/sarkisyan2016gfp/embeddings.txt'
    )

    X_train, y_train, X_test, y_test, mutations_test = split_X(
        X, meta
    )

    # Use brightness cutoff of 3 used in original study.
    y_train -= 3.
    y_test_cutoff = y_test - 3.
    y_train[y_train < 0.] = 0.
    y_test_cutoff[y_test_cutoff < 0.] = 0.

    y_unk_pred = np.load(
        'target/prediction_cache/gfp_cv_ypred_{}.npy'
        .format(model)
    )
    var_unk_pred = np.load(
        'target/prediction_cache/gfp_cv_varpred_{}.npy'
        .format(model)
    )
    acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta)

    n_surface, n_buried = 0, 0
    ns_surface, ns_buried = [], []
    acq_argsort = np.argsort(-acquisition)
    pos2counts = {}
    for i, acq_idx in enumerate(acq_argsort):
        mutations = mutations_test[acq_idx].split(':')
        for mutation in mutations:
            start_aa = mutation[1]
            end_aa = mutation[-1]
            pos = int(mutation[2:-1])
            assert(ref_seq[pos] == start_aa)
            if pos in surface_idxs:
                n_surface += 1
            else:
                n_buried += 1
            if i < 100:
                if pos not in pos2counts:
                    pos2counts[pos] = 0
                pos2counts[pos] += 1
        if i == 49:
            print('{}, Surface: {}, buried: {}'
                  .format(i + 1, n_surface, n_buried))
        elif i == 4999:
            print('{}, Surface: {}, buried: {}'
                  .format(i + 1, n_surface, n_buried))
        ns_surface.append(n_surface)
        ns_buried.append(n_buried)

    print('Total, Surface: {}, buried: {}'
          .format(len(surface_idxs),
                  len(ref_seq) - len(surface_idxs)))

    plt.figure()
    plt.plot(np.array(range(len(acquisition))), ns_surface)
    plt.plot(np.array(range(len(acquisition))), ns_buried)
    plt.legend([ 'N surface', 'N buried' ])
    plt.savefig('figures/gfp_structure.png', dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(list(range(len(y_test))), y_test[acq_argsort],
                alpha=0.01, c='#008080')
    plt.title('Spearman r = {:.4g}, P = {:.4g}'.format(
        *ss.spearmanr(
            list(range(len(y_test))), y_test[acq_argsort]
        )
    ))
    plt.savefig('figures/gfp_corr_{}.png'.format(model), dpi=300)
    plt.close()

    if model in { 'gp', 'hybrid', 'linear' }:
        plt.figure()
        plt.scatter(y_unk_pred, np.log1p(var_unk_pred), c=y_test_cutoff,
                    cmap='viridis', alpha=0.3)
        plt.title('GFP {}'.format(model))
        plt.savefig('figures/gfp_ypred_var_{}.png'.format(model))
        plt.close()

    cmap = matplotlib.cm.get_cmap('viridis')
    max_count = float(max([ pos2counts[pos] for pos in pos2counts ]))
    with open('data/sarkisyan2016gfp/gfp_acquistion.pml', 'w') as of:
        for pos in range(len(ref_seq)):
            if pos in pos2counts:
                val = min(pos2counts[pos] / max_count, 0.5) / 0.5
            else:
                val = 0
            of.write('select toColor, resi {} and chain A\n'
                     .format(pos + 2))
            rgb = cmap(val)
            of.write('color {}, toColor\n'
                     .format(matplotlib.colors.rgb2hex(rgb))
                     .replace('#', '0x'))

if __name__ == '__main__':
    model = sys.argv[1]
    beta = float(sys.argv[2])

    if len(sys.argv) >= 4:
        seed = int(sys.argv[3])
        np.random.seed(seed)
    else:
        seed = 1

    gfp_cv(model, beta, seed)

    gfp_structure(model, beta, seed)
