"""
Small helpers: the theoretical sample size, choosing a truncation by energy, and error norms.
"""
import numpy as np


def c_delta(delta):
    """The constant c_delta = 1 / (delta + (1 - delta) log(1 - delta)) in the sample-size bound (about 6.52 for delta = 1/2)."""
    return 1.0 / (delta + (1 - delta) * np.log(1 - delta))


def sample_size(N, delta=0.5, eps=0.5):
    """
    Number of samples M = ceil(c_delta * N * log(2N / eps)).

    With M optimally drawn samples, ||G - I||_2 <= delta (so cond(G) <= (1+delta)/(1-delta))
    holds with probability at least 1 - eps. Here N is the dimension of the polynomial
    space P (the number of rows of the index set), not of the full operator space.
    """
    return int(np.ceil(c_delta(delta) * N * np.log(2 * N / eps)))


def energy_truncation(variances, fraction):
    """
    Smallest d such that the first d coordinates carry at least `fraction` of the total energy.

    For an input f with independent coefficients f_j, E||f||^2 = sum_j E[f_j^2], so pass
    the second moments (variances, for centered coefficients) in decreasing order of importance.
    """
    c = np.cumsum(np.asarray(variances, dtype=float))
    return int(np.searchsorted(c, fraction * c[-1]) + 1)


def bochner_error(pred, true):
    """Empirical Bochner (L^2_rho) error: sqrt( mean_i ||pred_i - true_i||^2 ), over the rows."""
    return np.sqrt(np.mean(np.sum(np.abs(pred - true) ** 2, axis=1)))


def relative_bochner_error(pred, true):
    """Relative empirical Bochner error: bochner_error(pred, true) / sqrt( mean_i ||true_i||^2 )."""
    return bochner_error(pred, true) / np.sqrt(np.mean(np.sum(np.abs(true) ** 2, axis=1)))
