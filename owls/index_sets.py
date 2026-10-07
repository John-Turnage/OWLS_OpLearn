"""
Multi-index sets Lambda that pick out which polynomials span the space P.

Each row of an index set is a multi-index lambda = (lambda_1, ..., lambda_d), and
stands for the d-variate polynomial

    p_lambda(f) = p_{lambda_1}(f_1) * p_{lambda_2}(f_2) * ... * p_{lambda_d}(f_d),

where f_j is the j-th input coefficient and p_n is the degree-n orthonormal
polynomial of the j-th input measure. The number of rows is N_eff = dim P.
"""
import warnings
from collections import deque

import numpy as np


def index_set(d, k, kind="hc", weights=None, max_degree=None, max_size=None, p=2):
    """
    Build a (possibly anisotropic) downward-closed index set in d dimensions.

    The weights w_j in (0, 1] make some directions "cheaper" than others:
    a smaller w_j allows lower polynomial degrees in direction j.

        kind = "hc" (hyperbolic cross):   prod_j (1 + lambda_j)^(1 / w_j)   <= k + 1
        kind = "lp" (weighted lp ball):   (sum_j lambda_j^(p / w_j))^(1/p)  <= k

    With all weights equal to 1 these are the usual hyperbolic cross and lp ball.
    Note that the weights enter as an exponent 1/w_j (not as a multiplier).

    Parameters
    ----------
    d : int
        Number of input coordinates.
    k : float
        Size parameter (larger k gives a bigger set).
    kind : {"hc", "lp"}
        Which criterion to use.
    weights : array of length d, optional
        Anisotropy weights in (0, 1]. Default: all ones. A weight of exactly 0
        is replaced by 1e-10, which switches that direction off.
    max_degree : int, optional
        Largest allowed degree in any single direction.
    max_size : int, optional
        Stop once this many indices have been found (a warning is issued).
        Indices are found in breadth-first order, so a truncated set is
        still downward closed, but it is no longer exactly the set above.
    p : float
        Exponent of the lp ball (ignored for "hc").

    Returns
    -------
    (N, d) int array of multi-indices, sorted lexicographically.
    """
    if kind not in ("hc", "lp"):
        raise ValueError(f"Unknown kind '{kind}': use 'hc' or 'lp'")
    if max_degree is None:
        max_degree = np.inf
    if max_size is None:
        max_size = np.inf

    if weights is None:
        weights = np.ones(d)
    weights = np.array(weights, dtype=float)
    if weights.shape[0] != d:
        raise ValueError("Length of the weight array must match the dimension d")
    weights[weights == 0] = 1e-10
    if not np.all((weights > 0) & (weights <= 1)):
        raise ValueError("All weights must lie in (0,1]")

    threshold = k + 1 if kind == "hc" else k

    found = set()
    visited = {(0,) * d}
    queue = deque([(0,) * d])
    truncated = False

    # Breadth-first search outward from the zero index. Because both criteria
    # increase with every lambda_j, an index that fails the test cannot have a
    # neighbor further out that passes, so we only expand indices that pass.
    while queue:
        current = queue.popleft()

        if kind == "hc":
            criterion = 1.0
            for i, n in enumerate(current):
                criterion *= (1 + n) ** (1.0 / weights[i])
        else:
            s = 0.0
            for i, n in enumerate(current):
                s += n ** (p / weights[i])
            criterion = s ** (1.0 / p)

        if criterion > threshold:
            continue

        if len(found) < max_size:
            found.add(current)
        else:
            truncated = True
            break

        for i in range(d):
            if current[i] < max_degree:
                neighbor = list(current)
                neighbor[i] += 1
                neighbor = tuple(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    if truncated:
        warnings.warn(f"index_set stopped at max_size={max_size}; the full set is larger.")

    return np.array(sorted(found), dtype=int)


def linear_index_set(n, d):
    """
    Index set for *linear* operators: the first n unit vectors e_1, ..., e_n in d dimensions.

    The polynomial for e_j is p_1(f_j) = (f_j - mean_j) / std_j, so the space
    spanned is the set of (affine) linear functions of the first n coefficients.
    """
    if n > d:
        raise ValueError("Need n <= d")
    return np.eye(n, d, dtype=int)
