"""
Input measures, their orthonormal polynomials, and sampling.

An input function f is represented by its coefficient vector f = (f_1, ..., f_d) in
a fixed orthonormal basis. The input measure rho is a *product* of 1-D probability
measures, one per coefficient: f_j ~ rho_j independently. For each rho_j we use the
orthonormal polynomials p_0 = 1, p_1, p_2, ... of rho_j.

Three 1-D measures are available:

    Jacobi(alpha, beta)   density proportional to (1-x)^alpha (1+x)^beta on [-1, 1]
                          (an affinely mapped Beta(beta+1, alpha+1) distribution)
    Gaussian(mean, std)   normal distribution (Hermite polynomials)
    Gamma(shape, scale)   gamma distribution on [0, inf) (Laguerre polynomials)

Anywhere a list of measures is expected you may also pass an array of Jacobi
parameters: shape (d, 2) for (alpha_j, beta_j), or shape (d,) for symmetric
Jacobi measures with alpha_j = beta_j.

Two ways to draw training inputs are provided:

    sample_rho      Monte Carlo: f ~ rho.
    sample_optimal  the optimal sampling measure for a polynomial space P,

                        d mu(f) = (1/N) sum_{lambda in Lambda} p_lambda(f)^2 d rho(f).

                    mu is a mixture: pick a multi-index lambda uniformly at random,
                    then draw each f_j from the 1-D "induced" measure
                    p_{lambda_j}(x)^2 d rho_j(x). Each induced measure is replaced by
                    its restriction to a Q-point Gauss quadrature rule of rho_j, which
                    makes sampling a simple table lookup (inverse CDF).
"""
from functools import lru_cache

import numpy as np

from ._vendor.families import JacobiPolynomials, HermitePolynomials, LaguerrePolynomials


def as_rng(rng=None):
    """Turn None, an int seed, a np.random.Generator or a np.random.RandomState into a random generator."""
    if rng is None or isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    if isinstance(rng, (np.random.Generator, np.random.RandomState)):
        return rng
    raise TypeError("rng must be None, an int, a np.random.Generator or a np.random.RandomState")


def _christoffel_weights(ab, z):
    """
    Gauss weights w_q = 1 / sum_{j=0}^{Q-1} p_j(z_q)^2 at the Q Gauss nodes z, from the
    recurrence coefficients ab (same convention as owls._vendor.opoly1d). The polynomials
    are rescaled as they grow so nothing overflows; far-tail weights correctly underflow to 0.
    """
    Q = z.size
    p_prev = np.zeros_like(z)
    p = np.full_like(z, 1 / ab[0, 1])
    S = p**2
    log_scale = np.zeros_like(z)  # true sum = S * exp(log_scale)
    for j in range(1, Q):
        p_prev, p = p, ((z - ab[j, 0]) * p - ab[j - 1, 1] * p_prev) / ab[j, 1]
        S += p**2
        big = S > 1e100
        if np.any(big):
            f = 1 / np.sqrt(S[big])
            p[big] *= f
            p_prev[big] *= f
            log_scale[big] += np.log(S[big])
            S[big] = 1.0
    return np.exp(-log_scale) / S


# ---------------------------------------------------------------------------------------
# 1-D measures
# ---------------------------------------------------------------------------------------
class Measure1D:
    """
    Base class for a 1-D probability measure with orthonormal polynomials.

    Subclasses wrap one of the polynomial families in owls._vendor and an affine
    change of variables x = shift + scale * z, where z is the family's standard variable.
    """

    def _family(self):
        """A fresh polynomial family object in the standard variable z."""
        raise NotImplementedError

    # For measures with unbounded support the textbook (Golub-Welsch) Gauss weights are
    # only accurate in an absolute sense; the tiny weights far out in the tail come out as
    # rounding noise, which ruins quantities like w_q p_n(x_q)^2. Such measures recompute
    # the weights from the (relatively accurate) Christoffel sum w_q = 1 / sum_j p_j(x_q)^2.
    _accurate_tail_weights = False

    def _gauss(self, family, Q):
        """Q-point Gauss rule (standard variable z, weights) using the given family object."""
        z, w = family.gauss_quadrature(Q)
        if self._accurate_tail_weights:
            w = _christoffel_weights(family.recurrence(Q + 1), z)
        return z, w

    def _to_standard(self, x):
        return x

    def _from_standard(self, z):
        return z

    def eval(self, x, degrees):
        """Orthonormal polynomials p_n(x) for n in degrees, as an array of shape (len(x), len(degrees))."""
        z = self._to_standard(np.asarray(x, dtype=float))
        return self._family().eval(z, np.asarray(degrees))

    def quadrature(self, Q):
        """Q-point Gauss quadrature rule (nodes, weights) for this measure; the weights sum to 1."""
        z, w = self._gauss(self._family(), Q)
        return self._from_standard(z), w

    def sample(self, M, rng=None):
        """M independent samples from the measure."""
        raise NotImplementedError

    # Measures are compared (and cached) by their parameters.
    def _key(self):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self):
        return hash((type(self).__name__, self._key()))

    def __repr__(self):
        return f"{type(self).__name__}{self._key()}"


class Jacobi(Measure1D):
    """
    Jacobi probability measure on [-1, 1] with density proportional to (1-x)^alpha (1+x)^beta.

    For the symmetric case alpha = beta this has mean 0 and variance 1/(2 alpha + 3).
    alpha = beta = 0 is the uniform distribution.
    """

    def __init__(self, alpha, beta=None):
        beta = alpha if beta is None else beta
        if not (alpha > -1 and beta > -1):
            raise ValueError("Jacobi parameters must satisfy alpha, beta > -1")
        self.alpha, self.beta = float(alpha), float(beta)

    def _family(self):
        return JacobiPolynomials(self.alpha, self.beta)

    def eval(self, x, degrees):
        # eval_1d is the routine used to produce the paper's results.
        return self._family().eval_1d(np.asarray(x, dtype=float), np.asarray(degrees))

    def sample(self, M, rng=None):
        # If u ~ Beta(alpha+1, beta+1) then 1 - 2u has density ~ (1-x)^alpha (1+x)^beta.
        return 1 - 2 * as_rng(rng).beta(self.alpha + 1, self.beta + 1, M)

    @property
    def mean(self):
        return (self.beta - self.alpha) / (self.alpha + self.beta + 2)

    @property
    def std(self):
        a, b = self.alpha, self.beta
        return np.sqrt(4 * (a + 1) * (b + 1) / ((a + b + 2) ** 2 * (a + b + 3)))

    def _key(self):
        return (self.alpha, self.beta)


class Gaussian(Measure1D):
    """Normal distribution N(mean, std^2), with orthonormal Hermite polynomials."""

    def __init__(self, mean=0.0, std=1.0):
        if not std > 0:
            raise ValueError("std must be positive")
        self.mean, self.std = float(mean), float(std)

    _accurate_tail_weights = True

    # The vendored Hermite family is orthonormal for the density exp(-z^2)/sqrt(pi),
    # i.e. N(0, 1/2). So x = mean + sqrt(2) * std * z.
    def _family(self):
        return HermitePolynomials()

    def _to_standard(self, x):
        return (x - self.mean) / (np.sqrt(2) * self.std)

    def _from_standard(self, z):
        return self.mean + np.sqrt(2) * self.std * z

    def sample(self, M, rng=None):
        return as_rng(rng).normal(self.mean, self.std, M)

    def _key(self):
        return (self.mean, self.std)


class Gamma(Measure1D):
    """Gamma distribution with density proportional to x^(shape-1) exp(-x/scale) on [0, inf), with Laguerre polynomials."""

    def __init__(self, shape=1.0, scale=1.0):
        if not (shape > 0 and scale > 0):
            raise ValueError("shape and scale must be positive")
        self.shape, self.scale = float(shape), float(scale)

    _accurate_tail_weights = True

    # The vendored Laguerre family is orthonormal for z^rho exp(-z) / Gamma(rho+1).
    def _family(self):
        return LaguerrePolynomials(rho=self.shape - 1)

    def _to_standard(self, x):
        return x / self.scale

    def _from_standard(self, z):
        return self.scale * z

    def sample(self, M, rng=None):
        return as_rng(rng).gamma(self.shape, self.scale, M)

    @property
    def mean(self):
        return self.shape * self.scale

    @property
    def std(self):
        return np.sqrt(self.shape) * self.scale

    def _key(self):
        return (self.shape, self.scale)


def as_measures(params, d=None):
    """
    Convert input-measure parameters to a list of 1-D measures.

    params can be a list of Measure1D objects, a (d, 2) array of Jacobi (alpha, beta)
    pairs, or a length-d array of symmetric Jacobi parameters alpha (beta = alpha).
    """
    if isinstance(params, Measure1D):
        raise TypeError("Pass a list of measures, one per input coordinate")
    if len(params) > 0 and all(isinstance(m, Measure1D) for m in params):
        measures = list(params)
    else:
        arr = np.asarray(params, dtype=float)
        if arr.ndim == 1:
            measures = [Jacobi(a, a) for a in arr]
        elif arr.ndim == 2 and arr.shape[1] == 2:
            measures = [Jacobi(a, b) for a, b in arr]
        else:
            raise ValueError("Jacobi parameters must have shape (d,) or (d, 2)")
    if d is not None and len(measures) != d:
        raise ValueError(f"Expected {d} input measures, got {len(measures)}")
    return measures


def alpha_from_variance(var):
    """Symmetric Jacobi parameter alpha whose measure has the given variance: var = 1/(2 alpha + 3)."""
    return ((1 / np.asarray(var)) - 3) / 2


# ---------------------------------------------------------------------------------------
# Polynomial features
# ---------------------------------------------------------------------------------------
def vandermonde(x, index_set, measures):
    """
    Evaluate the tensor-product polynomials at the rows of x.

        V[i, n] = prod_j p^{(j)}_{Lambda[n, j]}(x[i, j])

    Parameters
    ----------
    x : (M, d) array of input coefficients.
    index_set : (N, d) int array of multi-indices Lambda.
    measures : list of d 1-D measures (or Jacobi parameters, see as_measures).

    Returns
    -------
    V : (M, N) array.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    L = np.asarray(index_set).astype(int)
    M, d = x.shape
    measures = as_measures(measures, d)
    if L.shape[1] != d:
        raise ValueError("index_set and x must have the same number of columns")

    V = np.ones((M, L.shape[0]))
    # Only directions with a nonzero degree change a column, since p_0 = 1.
    for j in range(d):
        cols = np.flatnonzero(L[:, j])
        if cols.size == 0:
            continue
        degs = L[cols, j]
        P = measures[j].eval(x[:, j], np.arange(degs.max() + 1))
        V[:, cols] *= P[:, degs]
    return V


# ---------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------
def sample_rho(M, measures, rng=None):
    """Draw M Monte Carlo samples f ~ rho (independent coordinates). Returns an (M, d) array."""
    rng = as_rng(rng)
    measures = as_measures(measures)
    samples = np.empty((M, len(measures)))
    for j, m in enumerate(measures):
        samples[:, j] = m.sample(M, rng)
    return samples


@lru_cache(maxsize=512)
def induced_cdf(measure, max_degree, n_quad=1000):
    """
    Discrete induced distributions of a 1-D measure, on its n_quad-point Gauss rule.

    Returns (nodes, F), where F[q, n] = sum_{i <= q} w_i p_n(x_i)^2 is the CDF of the
    degree-n induced measure p_n^2 d rho evaluated at node q (n = 0, ..., max_degree).
    Results are cached, since many input coordinates often share the same measure.
    """
    if max_degree >= n_quad:
        raise ValueError("Need max_degree < n_quad for the quadrature to be exact")
    family = measure._family()
    z, w = measure._gauss(family, n_quad)
    P = family.eval(z, range(max_degree + 1))
    W = np.tile(np.sqrt(w), [max_degree + 1, 1]).T * P
    F = np.cumsum(W**2, axis=0)
    nodes = measure._from_standard(z)
    nodes.setflags(write=False)
    F.setflags(write=False)
    return nodes, F


def sample_optimal(M, index_set, measures, rng=None, n_quad=1000):
    """
    Draw M samples from the optimal sampling measure of span{p_lambda : lambda in index_set}.

        d mu(f) = (1/N) sum_lambda p_lambda(f)^2 d rho(f)

    For each sample: pick a row lambda of the index set uniformly at random, then draw
    each coordinate f_j from the induced measure p_{lambda_j}^2 d rho_j, discretized on an
    n_quad-point Gauss rule (inverse-CDF lookup).

    Returns an (M, d) array.
    """
    rng = as_rng(rng)
    L = np.asarray(index_set).astype(int)
    N, d = L.shape
    measures = as_measures(measures, d)
    max_degree = int(L.max())

    # Pick the multi-index for each sample (rows are unique, so this is uniform over Lambda).
    rows, counts = np.unique(L, axis=0, return_counts=True)
    picked = rows[rng.choice(np.arange(rows.shape[0]), M, p=counts / N)]
    U = rng.uniform(0, 1, [M, d])

    samples = np.empty((M, d))
    for j in range(d):
        nodes, F = induced_cdf(measures[j], max_degree, n_quad)
        degs = picked[:, j]
        for n in np.unique(degs):
            idx = np.flatnonzero(degs == n)
            # Smallest node q with U < F[q, n]; clip guards against F[-1, n] being a hair below 1.
            q = np.searchsorted(F[:, n], U[idx, j], side="right")
            samples[idx, j] = nodes[np.minimum(q, n_quad - 1)]
    return samples
