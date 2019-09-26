import numpy as np

class HybridMLPEnsembleGP(object):
    def __init__(self, mlp_ensemble, gaussian_process):
        self.mlp_ensemble_ = mlp_ensemble
        self.gaussian_process_ = gaussian_process

    def fit(self, X, y):
        self.mlp_ensemble_.fit(X, y)

        y_pred = self.mlp_ensemble_.predict(X)

        self.gaussian_process_.fit(X, y - y_pred)

        #X_tiled = np.tile(X, (self.mlp_ensemble_.n_regressors_, 1))
        #y_tiled = np.tile(y.flatten(), self.mlp_ensemble_.n_regressors_)
        #y_pred_tiled = self.mlp_ensemble_.multi_predict_.flatten('F')

        #self.gaussian_process_.fit(X_tiled, y_tiled - y_pred_tiled)

    def predict(self, X):
        residual = self.gaussian_process_.predict(X)
        self.uncertainties_ = self.gaussian_process_.uncertainties_

        return self.mlp_ensemble_.predict(X) + residual
