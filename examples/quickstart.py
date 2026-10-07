"""
Quickstart: learn a nonlinear solution operator with optimally weighted least squares.

The operator. For a forcing f on (0, 1), let u = K(f) solve the nonlinear-in-f problem

    -u''(x) = exp(f(x)) - 1,    u(0) = u(1) = 0.

Inputs and outputs are written in the sine basis phi_k(x) = sqrt(2) sin(k pi x):

    f = sum_{j=1}^{6} f_j phi_j,         f_j ~ Jacobi(j, j) independently on [-1, 1],
    u = sum_{k=1}^{32} u_k phi_k,        u_k = <exp(f) - 1, phi_k> / (k pi)^2.

We approximate K by polynomials of total degree <= p in (f_1, ..., f_6), fit from
M = owls.sample_size(N) samples, and compare optimal sampling with plain Monte Carlo.

Run with:  python examples/quickstart.py
"""
import numpy as np

import owls

D_IN, D_OUT = 6, 32

# --- 1. Input measure: one 1-D measure per input coefficient --------------------------------
measures = [owls.Jacobi(j, j) for j in range(1, D_IN + 1)]

# --- 2. The "solver": input coefficients (M, D_IN) -> output coefficients (M, D_OUT) --------
xq, wq = np.polynomial.legendre.leggauss(256)          # Gauss-Legendre rule on [0, 1]
xq, wq = (xq + 1) / 2, wq / 2
phi_in = np.sqrt(2) * np.sin(np.pi * np.outer(xq, np.arange(1, D_IN + 1)))
phi_out = np.sqrt(2) * np.sin(np.pi * np.outer(xq, np.arange(1, D_OUT + 1)))


def solver(f_coeffs):
    rhs = np.exp(f_coeffs @ phi_in.T) - 1                # exp(f) - 1 at the quadrature nodes
    return (rhs * wq) @ phi_out / (np.pi * np.arange(1, D_OUT + 1)) ** 2


# --- 3-5. Index set, sampling and fitting ----------------------------------------------------
rng = np.random.default_rng(0)
f_test = owls.sample_rho(1000, measures, rng)           # test inputs from rho
u_test = solver(f_test)

print(f"{'degree':>6} {'N':>5} {'M':>6} | {'cond(G) optimal':>15} {'error optimal':>13} | {'cond(G) MC':>10} {'error MC':>9}")
for p in range(1, 7):
    index_set = owls.index_set(D_IN, p, kind="lp", p=1)   # total degree <= p
    row = f"{p:6d} {len(index_set):5d} {owls.sample_size(len(index_set)):6d} |"
    for optimal in (True, False):
        model = owls.learn_operator(solver, measures, index_set, optimal=optimal, rng=rng)
        err = owls.relative_bochner_error(model.predict(f_test), u_test)
        row += f" {model.cond():15.2f} {err:13.2e} |" if optimal else f" {model.cond():10.2f} {err:9.2e}"
    print(row)

print("\nOptimal sampling keeps cond(G) below (1 + 1/2) / (1 - 1/2) = 3 with high probability;")
print("Monte Carlo sampling needs more samples as the degree grows.")
