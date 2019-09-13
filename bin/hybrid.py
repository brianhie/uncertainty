import multiprocessing

def regressor_fit(regressor, X, y):
    regressor.fit(X, y)

def regressor_predict(regressor, X, return_dict, prefix):
    mean, var = regressor.predict(X)
    return_dict['{}_mean'.format(prefix)] = mean
    return_dict['{}_var'.format(prefix)] = var

class HybridMLPEnsembleGP(object):
    def __init__(self, mlp_ensemble, gaussian_process):
        self.mlp_ensemble_ = mlp_ensemble
        self.gaussian_process_ = gaussian_process

    def fit(self, X, y):
        processes = [
            multiprocessing.Process(
                target=regressor_fit, args=(self.mlp_ensemble_, X, y)
            ),
            multiprocessing.Process(
                target=regressor_fit, args=(self.gaussian_process_, X, y)
            )
        ]

        [ proc.start() for proc in processes ]

        [ proc.join() for proc in processes ]

    def predict(self, X):
        return_dict = {}

        processes = [
            multiprocessing.Process(
                target=regressor_predict,
                args=(self.mlp_ensemble_, X, return_dict, 'mlp')
            ),
            multiprocessing.Process(
                target=regressor_predict,
                args=(self.gaussian_process_, X, return_dict, 'gp')
            )
        ]

        [ proc.start() for proc in processes ]

        [ proc.join() for proc in processes ]

        self.uncertainties_ = return_dict['gp_var']

        return return_dict['mlp_mean']
