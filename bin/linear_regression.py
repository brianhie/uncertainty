from utils import *

from sklearn.linear_model import LinearRegression

class LinearRegressor(object):
    def __init__(self):
        self.model_ = LinearRegression(
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            n_jobs=40
        )

    def fit(self, X, y):
        self.model_.fit(X, y)

    def predict(self, X):
        self.uncertainties_ = np.zeros(X.shape[0])
        return self.model_.predict(X)
