# Modified from sklearn GP example.
# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>s
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x) + x * np.cos(2 * x) + 10 * np.sin(x) #+ x * np.sin(3.1 * x + 2)

if __name__ == '__main__':
    X = np.atleast_2d(np.linspace(10, 20, 80)).T
    #X = np.atleast_2d([ 5.2, 6., 7., 8.7, 10., 11., 15.5, 16]).T

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x = np.atleast_2d(np.linspace(0, 20, 1000)).T

    plt.figure()


    for pidx in range(4):
        if pidx == 3:
            X = np.vstack([ X, np.array([4.8, 5.5]).reshape(-1, 1) ])

        y = f(X).ravel()

        if pidx == 1:
            dy = 1
        if pidx >= 2:
            dy = 10 + 1.0 * np.random.random(y.shape)
            noise = np.random.normal(0, dy)
            y += noise

        if pidx == 0:
            gp = MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, alpha=0, solver='adam', activation='logistic')
        else:
            # Instantiate a Gaussian Process model
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                          n_restarts_optimizer=10)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X, y)

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        if pidx == 0:
            y_pred = gp.predict(x)
        else:
            y_pred, sigma = gp.predict(x, return_std=True)

        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.subplot(4, 1, pidx + 1)
        plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
        plt.plot(x, y_pred, '#A5D1EC', label='Prediction')
        if pidx == 1:
            plt.fill(np.concatenate([x, x[::-1]]),
                     np.concatenate([y_pred - sigma,
                                     (y_pred + sigma)[::-1]]),
                     alpha=0.3, fc='#90D9EE', ec='None', label='95% confidence interval')
        elif pidx >= 2:
            plt.fill(np.concatenate([x, x[::-1]]),
                     np.concatenate([y_pred - 1.5 * sigma,
                                     (y_pred + 1.5 * sigma)[::-1]]),
                     alpha=0.3, fc='#90D9EE', ec='None', label='95% confidence interval')
        plt.scatter(X.ravel(), y, color='#011B56', label='Observations', s=3,)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-60, 60)

    plt.legend(loc='upper left')
    plt.savefig('figures/uncertainty_cartoon.svg')
    plt.close()
