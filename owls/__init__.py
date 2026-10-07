"""
owls: optimally weighted least squares for operator learning.

The method in five steps:

  1. Encode inputs f by their coefficients (f_1, ..., f_d) in an orthonormal basis, and
     put a product probability measure rho on them, one 1-D measure per coefficient
     (owls.Jacobi, owls.Gaussian, owls.Gamma).
  2. Encode outputs g = K(f) by their coefficients in an orthonormal basis of the
     output space (any basis you like; the method does not care how many).
  3. Pick a polynomial space P in the input coefficients with an index set
     (owls.index_set, owls.linear_index_set). Its size N = dim P controls everything.
  4. Draw M training inputs from the optimal sampling measure (owls.sample_optimal),
     with M about c N log N (owls.sample_size), and run your solver on them.
  5. Fit by weighted least squares (owls.BlockWLS) and check the Gram matrix
     (model.cond(), model.gram_deviation()).

Steps 4 and 5 in one call: owls.learn_operator(solver, measures, index_set).
For a fixed dataset that you cannot sample from freely, use owls.EmpiricalWLS.

Conventions
-----------
- measures: a list of 1-D measures, one per input coefficient. An array of Jacobi
  parameters is also accepted: shape (d, 2) for (alpha_j, beta_j), or shape (d,) for
  symmetric Jacobi measures (beta_j = alpha_j).
- Polynomials are orthonormal for the probability measure: p_0 = 1 and
  p_1(x) = (x - mean) / std.
- index sets are (N, d) integer arrays, one multi-index per row.
- Inputs are (M, d) arrays and outputs (M, d_out) arrays (outputs may be complex).
- Functions that use randomness take rng: None, an int seed, a np.random.Generator,
  or a np.random.RandomState.
"""
from .index_sets import index_set, linear_index_set
from .measures import (
    Jacobi, Gaussian, Gamma, as_measures, alpha_from_variance,
    vandermonde, sample_rho, sample_optimal, induced_cdf,
)
from .wls import BlockWLS, EmpiricalWLS, learn_operator, gram_cond, gram_deviation
from .utils import c_delta, sample_size, energy_truncation, bochner_error, relative_bochner_error

__version__ = "1.0.0"

__all__ = [
    "index_set", "linear_index_set",
    "Jacobi", "Gaussian", "Gamma", "as_measures", "alpha_from_variance",
    "vandermonde", "sample_rho", "sample_optimal", "induced_cdf",
    "BlockWLS", "EmpiricalWLS", "learn_operator", "gram_cond", "gram_deviation",
    "c_delta", "sample_size", "energy_truncation", "bochner_error", "relative_bochner_error",
]
