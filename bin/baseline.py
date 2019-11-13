import numpy as np

class Baseline(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        self.uncertainties_ = np.zeros(X.shape[0])
        return np.zeros(X.shape[0])
