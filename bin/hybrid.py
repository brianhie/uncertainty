import multiprocessing

class HybridMLPEnsembleGP(object):
    def __init__(self, mlp_ensemble, gaussian_process):
        self.mlp_ensemble_ = mlp_ensemble
        self.gaussian_process_ = gaussian_process

    def fit(self, X, y):
        self.mlp_ensemble_.fit(X, y)

        y_pred = self.mlp_ensemble_.predict(X)
        self.gaussian_process_.fit(X, y - y_pred)

    def predict(self, X):
        residual = self.gaussian_process_.predict(X)
        self.uncertainties_ = self.gaussian_process_.uncertainties_

        return self.mlp_ensemble_.predict(X) + residual
