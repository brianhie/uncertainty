# Modified from sklearn GP example.
# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>s
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x):
    """The function to predict."""
    return 10 * np.sin(x)

if __name__ == '__main__':
    X = np.atleast_2d([ 5.2, 6., 7., 8.7, 8.7, 10., 11., 11.7, 15.5, 16]).T

    # Observations and noise
    y = f(X).ravel()
    dy = 0.4 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x = np.atleast_2d(np.linspace(3, 16.5, 1000)).T

    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                  n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.figure()
    #plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(x, y_pred, '#0269A4', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=0.3, fc='#90D9EE', ec='None', label='95% confidence interval')
    plt.scatter(X.ravel(), y, color='#011B56', label='Observations')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-40, 40)
    plt.legend(loc='upper left')
    plt.savefig('figures/gp_cartoon.svg')
    plt.close()
