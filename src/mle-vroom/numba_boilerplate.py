import numpy as np
from numba import njit

@njit
def normal_loglik_numba(x, mu, sigma):
    n = x.size
    var = sigma * sigma
    return -0.5 * n * np.log(2 * np.pi * var) - np.sum((x - mu)**2) / (2 * var)
