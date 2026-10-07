"""
Viscous Burgers' equation solver used to generate data for the Burgers experiments.

    u_t + u u_x = nu u_xx,   x in (0, 1),   u(0, t) = u(1, t) = 0,

solved by a sine spectral Galerkin method,

    u(x, t) = sum_{j=1}^{N} U_j(t) sqrt(2) sin(j pi x),

with first-order IMEX Euler time stepping: the viscous term is implicit (it is
diagonal in the sine basis) and the nonlinear term u u_x is explicit, evaluated
pseudo-spectrally on an equispaced grid of 4N+1 points (enough to avoid aliasing).
"""
import numpy as np


def sine_basis_vandermonde(x, N):
    """sin(j pi x) for j = 1..N, as an x.size x N array."""
    assert N > 0, "N must be a positive integer"
    return np.sin(np.pi * np.outer(x.flatten(), np.arange(1, N + 1)))


def cosine_basis_vandermonde(x, N):
    """cos(j pi x) for j = 0..N, as an x.size x (N+1) array."""
    assert N > 0, "N must be a positive integer"
    return np.cos(np.pi * np.outer(x.flatten(), np.arange(N + 1)))


def fourier_quad_rule(M, interval=[0, 1], prob_measure=False):
    """M-point equispaced, equally weighted quadrature rule on [a, b) (weights sum to 1 if prob_measure)."""
    assert M > 0, "M must be a positive integer"
    x = np.linspace(interval[0], interval[1], M + 1)[:M]
    w = np.ones(M) / M
    if not prob_measure:
        w *= (interval[1] - interval[0])
    return x, w


def burgers_spectral_galerkin(U0, nu, dt, n_steps):
    """
    Advance sine-Galerkin coefficients of Burgers' equation by n_steps steps of size dt.

    U0 : (K, N) array of initial coefficients U_j(0) (one initial condition per row),
         or a length-N vector. N, the number of modes in the solve, is taken from U0.
    Returns the (K, N) coefficients at time T = n_steps * dt.
    """
    if len(U0.shape) == 1:
        N = U0.size
        K = 1
    elif len(U0.shape) == 2:
        K, N = U0.shape

    U = np.zeros([K, N])
    U[:, :] = U0

    # The update for mode j is
    #   U_j^{n+1} (1 + dt nu (j pi)^2) = U_j^n - dt (u u_x)_j
    d2invmat = 1 / (1 + dt * nu * (np.pi * np.arange(1, N + 1)) ** 2)

    # Grid values of u and u_x, and the projection back onto the sine basis
    x, w = fourier_quad_rule(4 * N + 1, interval=[0, 1])
    d1_nodal_eval = np.sqrt(2) * cosine_basis_vandermonde(x, N)[:, 1:]
    d1_nodal_eval *= np.pi * np.arange(1, N + 1)
    V = np.sqrt(2) * sine_basis_vandermonde(x, N)
    wV = (V.T * w).T

    for m in range(n_steps):
        U = (U - dt * ((U @ V.T) * (U @ d1_nodal_eval.T)) @ wV) * d2invmat

    return U


def burgers_map(U0, nu, T, dt, d_solve, d_out=None):
    """
    The operator u(., 0) -> u(., T) on sine coefficients.

    The d_in = U0.shape[1] input coefficients are zero-padded to d_solve modes,
    solved to time T with time step dt, and the first d_out output coefficients are
    returned (all d_solve if d_out is None). Raises an error if the solve blew up.
    """
    U0 = np.atleast_2d(U0)
    lifted = np.zeros([U0.shape[0], d_solve])
    lifted[:, :U0.shape[1]] = U0
    out = burgers_spectral_galerkin(lifted, nu, dt, int(T / dt))
    if d_out is not None:
        out = out[:, :d_out]
    if not np.all(np.isfinite(np.linalg.norm(out, axis=1))):
        raise ValueError("Encountered inf/nan while solving the PDE. Probably you want to decrease dt.")
    return out
