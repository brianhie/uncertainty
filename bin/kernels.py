from sklearn.gaussian_process.kernels import *

def _check_length_scale(X, length_scale, length_scale_dims):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if len(length_scale) != len(length_scale_dims):
        raise ValueError("length_scale must have same number of entries as "
                         "length_scale_dim")
    if sum(length_scale_dims) != X.shape[1]:
        raise ValueError("length_scale_dim must sum up to feature dimension")
    return length_scale


class FactorizedRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, length_scale, length_scale_dims,
                 length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_dims = length_scale_dims
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric",
                              self.length_scale_bounds,
                              len(self.length_scale))

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(
            X, self.length_scale, self.length_scale_dims)

        length_scale_vec = np.ones(X.shape[1])
        for i in range(len(self.length_scale)):
            start = sum(self.length_scale_dims[:i])
            end = sum(self.length_scale_dims[:i+1])
            length_scale_vec[start:end] *= length_scale[i]

        if Y is None:
            dists = pdist(X / length_scale_vec, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale_vec, Y / length_scale_vec,
                          metric='sqeuclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # We need to recompute the pairwise dimension-wise distances.
                K_gradient = np.empty((X.shape[0], X.shape[0],
                                       len(self.length_scale)))
                for i in range(len(self.length_scale)):
                    start = sum(self.length_scale_dims[:i])
                    end = sum(self.length_scale_dims[:i+1])
                    K_gradient[:, :, i] = (
                        K * squareform(pdist(X[:, start:end] / length_scale[i],
                                             metric='sqeuclidean'))
                    )
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        return "{0}(length_scale=[{1}])".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
