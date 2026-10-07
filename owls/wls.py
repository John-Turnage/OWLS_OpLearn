"""
Weighted least-squares (WLS) operator learning.

The operator K maps input coefficients f (length d) to output coefficients g = K(f)
(length d_out) in an orthonormal basis psi_1, ..., psi_{d_out} of the output space.
We approximate K in the tensor-product space V = Y_h (x) P, i.e.

    K(f)  ~  sum_{m=1}^{d_out}  sum_{n=1}^{N}  C[n, m] p_{lambda_n}(f) psi_m,

where p_{lambda_n} are the orthonormal polynomials picked out by an index set
(see owls.index_sets) and C is an N x d_out coefficient matrix.

Given training pairs (f_i, g_i), i = 1..M, the coefficients solve

    min_C  (1/M) sum_i  w(f_i) || g_i - C^T p(f_i) ||^2,

with weights

    w(f) = N / sum_n p_{lambda_n}(f)^2   if the f_i were drawn from the optimal measure mu,
    w(f) = 1                              if the f_i were drawn from rho (Monte Carlo).

Because the output basis is orthonormal, this splits into d_out independent
least-squares problems that all share the same M x N matrix
A = diag(sqrt(w/M)) V, with V[i, n] = p_{lambda_n}(f_i). In particular the Gram
matrix G = A^T A is only N x N, and its conditioning (and so the number of samples
needed for stability) depends on N = dim P, not on d_out.

Stability: if ||G - I||_2 <= delta, the discrete least-squares norm is within a
factor (1 +- delta) of the true L^2_rho norm on V, and cond(G) <= (1+delta)/(1-delta).
Under optimal sampling this holds with probability >= 1 - eps once
M >= c_delta N log(2N/eps); see owls.utils.sample_size.
"""
import warnings

import numpy as np

from .measures import Jacobi, as_measures, as_rng, vandermonde, sample_optimal, sample_rho


def _check_bool(optimal):
    if not isinstance(optimal, (bool, np.bool_)):
        raise TypeError("optimal must be True (optimal sampling and weights) or False (Monte Carlo, unit weights)")


def optimal_weights(V):
    """Weights w_i = N / sum_n V[i, n]^2 for samples drawn from the optimal measure."""
    return V.shape[1] / np.sum(V**2, axis=1)


def gram_cond(A):
    """Condition number of G = A^T A (infinite if A has fewer rows than columns)."""
    if A.shape[0] < A.shape[1]:
        return np.inf
    return np.linalg.cond(A) ** 2


def gram_deviation(A):
    """||G - I||_2 for G = A^T A, computed from the singular values of A."""
    s = np.linalg.svd(A, compute_uv=False)
    lam_max = s[0] ** 2
    lam_min = s[-1] ** 2 if A.shape[0] >= A.shape[1] else 0.0
    return max(abs(lam_max - 1), abs(1 - lam_min))


class _GramMixin:
    """cond() and gram_deviation() for classes that store the weighted matrix A."""

    def gram(self):
        """The N x N Gram matrix G = A^T A."""
        return self.A.T @ self.A

    def cond(self):
        """Condition number of the Gram matrix G."""
        return gram_cond(self.A)

    def gram_deviation(self, delta=0.5, warn=True):
        """
        ||G - I||_2, the quantity the stability theory controls.

        Warns if it exceeds delta (then the least-squares fit may be unreliable:
        take more samples, or use optimal sampling).
        """
        dev = gram_deviation(self.A)
        if warn and dev > delta:
            warnings.warn(f"||G - I||_2 = {dev:.3g} > delta = {delta}: the least-squares problem may be unstable.")
        return dev


class BlockWLS(_GramMixin):
    """
    WLS operator learning with inputs drawn from a known product measure rho.

    Example
    -------
        model = BlockWLS(index_set, measures).fit(x_train, y_train, optimal=True)
        y_pred = model.predict(x_test)
        model.cond()             # condition number of the Gram matrix

    Parameters
    ----------
    index_set : (N, d) int array
        Multi-indices of the polynomials spanning P.
    measures : list of d 1-D measures (or Jacobi parameters, see owls.measures.as_measures).
    """

    def __init__(self, index_set, measures):
        self.index_set = np.asarray(index_set).astype(int)
        self.N, self.d = self.index_set.shape
        self.measures = as_measures(measures, self.d)
        self.C = None

    def vandermonde(self, x):
        """V[i, n] = p_{lambda_n}(x_i)."""
        return vandermonde(x, self.index_set, self.measures)

    def fit(self, x, y, optimal=True):
        """
        Fit the coefficients C from training pairs.

        x : (M, d) training inputs. Draw them with owls.sample_optimal if optimal=True,
            or with owls.sample_rho if optimal=False.
        y : (M, d_out) training outputs (real or complex), in an orthonormal output basis.
        optimal : True for optimal-measure weights, False for unit (Monte Carlo) weights.
        """
        _check_bool(optimal)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        V = self.vandermonde(x)
        self.M = V.shape[0]
        self.optimal = optimal
        self.weights = optimal_weights(V) if optimal else np.ones(self.M)
        D = np.sqrt(self.weights / self.M)[:, np.newaxis]
        self.A = D * V
        self.C = np.linalg.lstsq(self.A, D * y, rcond=None)[0]
        return self

    def predict(self, x):
        """Apply the learned operator: returns the (M_new, d_out) predicted output coefficients."""
        if self.C is None:
            raise RuntimeError("Call fit() before predict()")
        return self.vandermonde(x) @ self.C

    def as_matrix(self):
        """
        For a linear index set (rows are unit vectors e_j), return the learned operator
        as a d_out x d matrix K with K[m, j] = <K_tilde xi_j, psi_m>.

        Since p_1(x) = (x - mean_j) / std_j, this is C divided row-wise by std_j
        (if the means are not zero it is the linear part of an affine map).
        """
        L = self.index_set
        if not (np.all(L.sum(axis=1) == 1) and np.all(L.max(axis=1) == 1)):
            raise ValueError("as_matrix() only applies to linear index sets (rows e_j)")
        j_of_row = L.argmax(axis=1)
        std = np.array([m.std for m in self.measures])
        K = np.zeros((self.C.shape[1], self.d), dtype=self.C.dtype)
        K[:, j_of_row] = (self.C / std[j_of_row, None]).T
        return K


class EmpiricalWLS(_GramMixin):
    """
    WLS operator learning from a fixed dataset ("pool") of input-output pairs.

    Use this when you cannot choose where the inputs are: the input measure is the
    empirical (uniform discrete) measure on the S pool inputs. The polynomials
    p_lambda are generally not orthonormal for this measure, so we orthonormalize them
    with a QR factorization of the S x N matrix V / sqrt(S) = Q R. The functions
    b(x) = V(x) R^{-1} are then orthonormal on the pool, and the optimal sampling
    measure is the discrete distribution

        Pr{pick pool point i} = ||Q[i, :]||^2 / N    (the leverage scores).

    Example
    -------
        model = EmpiricalWLS(x_pool, y_pool, index_set, measures)
        model.fit(M, optimal=True, rng=0)
        y_pred = model.predict(x_test)

    The measures only choose which polynomial family is used (e.g. Jacobi polynomials
    for inputs scaled to [-1, 1]); any reasonable choice works because of the QR step.
    """

    def __init__(self, x_pool, y_pool, index_set, measures):
        self.x_pool = np.asarray(x_pool)
        self.y_pool = np.asarray(y_pool)
        self.index_set = np.asarray(index_set).astype(int)
        self.S, self.d = self.x_pool.shape
        self.N = self.index_set.shape[0]
        self.measures = as_measures(measures, self.d)
        self.C = None

        if all(isinstance(m, Jacobi) for m in self.measures):
            eps = 1e-10
            if not (np.all(self.x_pool >= -1.0 - eps) and np.all(self.x_pool <= 1.0 + eps)):
                raise ValueError("Inputs must be scaled to lie in [-1, 1] for Jacobi polynomials.")

        V = vandermonde(self.x_pool, self.index_set, self.measures) / np.sqrt(self.S)
        self.Q, self.R = np.linalg.qr(V)
        K = (self.S / self.N) * np.linalg.norm(self.Q, axis=1) ** 2
        self.leverage = K / np.sum(K)

    def fit(self, M, optimal=True, rng=None):
        """Draw M pool points (with replacement) and fit. optimal=False draws uniformly with unit weights."""
        _check_bool(optimal)
        rng = as_rng(rng)
        if optimal:
            self.sample_idx = rng.choice(np.arange(self.S), M, replace=True, p=self.leverage)
        else:
            self.sample_idx = rng.choice(np.arange(self.S), M, replace=True)
        self.M = M
        self.optimal = optimal

        B = np.sqrt(self.S) * self.Q[self.sample_idx, :]  # orthonormal basis at the sampled points
        self.weights = optimal_weights(B) if optimal else np.ones(M)
        sqrt_w = np.sqrt(self.weights / M)
        self.A = (B.T * sqrt_w).T
        # Right-hand side g = A^T diag(sqrt_w) y[sample_idx]. Pool points drawn more than once
        # are summed first, so the (possibly huge) M x d_out array y[sample_idx] is never formed.
        A_pool = np.zeros((self.S, self.N))
        np.add.at(A_pool, self.sample_idx, self.A * sqrt_w[:, None])
        g = A_pool.T @ self.y_pool
        self.C = np.linalg.solve(self.A.T @ self.A, g)
        return self

    def _basis(self, x):
        """Orthonormalized basis b(x) = V(x) R^{-1}."""
        x = np.asarray(x)
        n = x.shape[0]
        V = vandermonde(x, self.index_set, self.measures) / np.sqrt(n)
        return np.sqrt(n) * np.linalg.solve(self.R.T, V.T).T

    def predict(self, x):
        """Predicted output coefficients at new inputs x (same scaling as the pool)."""
        if self.C is None:
            raise RuntimeError("Call fit() before predict()")
        return self._basis(x) @ self.C

    def predict_on_fit_samples(self):
        """Predictions at the pool points used for the fit."""
        return np.sqrt(self.S) * self.Q[self.sample_idx, :] @ self.C

    def compute_errors(self, x, y):
        """Per-sample absolute and relative errors ||y_pred - y|| and ||y_pred - y|| / ||y||."""
        abs_err = np.linalg.norm(self.predict(x) - y, axis=1)
        return abs_err, abs_err / np.linalg.norm(y, axis=1)

    def compute_fit_sample_errors(self):
        """Per-sample absolute and relative errors on the pool points used for the fit."""
        y = self.y_pool[self.sample_idx]
        abs_err = np.linalg.norm(self.predict_on_fit_samples() - y, axis=1)
        return abs_err, abs_err / np.linalg.norm(y, axis=1)


def learn_operator(solver, measures, index_set, M=None, optimal=True, rng=None):
    """
    The whole pipeline in one call: sample inputs, run your solver, fit.

    Parameters
    ----------
    solver : function
        Maps an (M, d) array of input coefficients to an (M, d_out) array of output
        coefficients (in an orthonormal basis of the output space).
    measures : list of d 1-D input measures (or Jacobi parameters).
    index_set : (N, d) int array of multi-indices.
    M : int, optional
        Number of training samples. Default: owls.utils.sample_size(N), the number the
        theory asks for (delta = eps = 1/2).
    optimal : bool
        True: draw inputs from the optimal measure and use optimal weights.
        False: draw inputs from rho (Monte Carlo) with unit weights.
    rng : seed or random generator.

    Returns
    -------
    A fitted BlockWLS model. The training data are kept as model.x_train, model.y_train.
    """
    from .utils import sample_size

    _check_bool(optimal)
    rng = as_rng(rng)
    index_set = np.asarray(index_set).astype(int)
    measures = as_measures(measures, index_set.shape[1])
    if M is None:
        M = sample_size(index_set.shape[0])

    x = sample_optimal(M, index_set, measures, rng) if optimal else sample_rho(M, measures, rng)
    y = np.asarray(solver(x))
    model = BlockWLS(index_set, measures).fit(x, y, optimal=optimal)
    model.x_train, model.y_train = x, y
    return model
