import numpy as np
from scipy.optimize import minimize
from .numba_boilerplate import normal_loglik_numba

def mle_normal(x, use_numba=False):
    x = np.asarray(x)

    def neg_loglik(theta):
        mu, sigma = theta
        if sigma <= 0:
            return np.inf
        if use_numba:
            return -normal_loglik_numba(x, mu, sigma)
        else:
            n = len(x)
            var = sigma * sigma
            return 0.5 * n * np.log(2 * np.pi * var) + np.sum((x - mu)**2) / (2 * var)

    mu0 = np.mean(x)
    sigma0 = np.std(x)

    res = minimize(
        neg_loglik,
        x0=[mu0, sigma0],
        method="L-BFGS-B",
        bounds=[(None,None),(1e-6,None)]
    )
    return res.x
