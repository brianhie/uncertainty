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
        'mutations', 'n_mut', 'brightness'
    ])
    return X, meta

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
            n_epochs=300,
            seed=seed,
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
            n_epochs=300,
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
                n_epochs=300,
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

    regressor.fit(X_train, y_train)

    return regressor

def acquisition_rank(y_pred, var_pred, beta=1.):
    return ss.rankdata(y_pred) + (beta * ss.rankdata(-var_pred))


if __name__ == '__main__':
    model = sys.argv[1]
    beta = float(sys.argv[2])

    X, meta = load_embeddings(
        'data/sarkisyan2016gfp/embeddings.txt'
    )

    X_train, y_train, X_test, y_test, mutations_test = split_X(
        X, meta
    )

    # Use brightness cutoff of 3 used in original study.
    y_train -= 3.
    y_test -= 3.
    y_train[y_train < 0.] = 0.
    y_test[y_test < 0.] = 0.

    regressor = train(model, X_train, y_train)
    y_unk_pred = regressor.predict(X_test)
    var_unk_pred = regressor.uncertainties_

    acquisition = acquisition_rank(y_unk_pred, var_unk_pred, beta)

    acq_argsort = np.argsort(-acquisition)
    for rank, idx in enumerate(acq_argsort):
        fields = [
            rank, mutations_test[idx], y_test[idx]
        ]
        print('\t'.join([ str(field) for field in fields ]))
