# An Optimal Weighted Least-Squares Method for Operator Learning

Code for the paper

> J. Turnage, M. Lowery, J. Jakeman, Z. Morrow, A. Narayan, V. Shankar.
> *An Optimal Weighted Least-Squares Method for Operator Learning.*
> [arXiv:2512.11168](https://arxiv.org/abs/2512.11168)

The repository has two parts:

- **`owls/`**, a small, problem-independent Python package that implements the method.
- **`experiments/`**, notebooks that reproduce the Poisson, viscous Burgers' and Navier–Stokes results in the paper.

## The method in brief

We want to learn an operator $K: \mathcal X \to \mathcal Y$ between function spaces from input–output pairs $(f_i, K(f_i))$, where each output comes from an (expensive) solver.

1. **Inputs.** Write $f$ by its coefficients $(f_1, \dots, f_d)$ in an orthonormal basis. Choose a probability measure $\rho$ on them, with independent coordinates $f_j \sim \rho_j$. This measure is what accuracy is measured against.
2. **Outputs.** Write $K(f)$ by its coefficients in any orthonormal basis $\psi_1, \dots, \psi_{d_{\mathrm{out}}}$ of the output space.
3. **Approximation space.** Use operators of the form $\sum_{m} \sum_{\lambda \in \Lambda} c_{\lambda m}\, p_\lambda(f)\, \psi_m$, where $p_\lambda(f) = \prod_j p_{\lambda_j}(f_j)$ are orthonormal polynomials and $\Lambda$ is a multi-index set of size $N$. Linear operators are the special case where every $\lambda$ is a unit vector.
4. **Sampling.** Draw the training inputs from the *optimal sampling measure* $d\mu = \frac1N \sum_{\lambda\in\Lambda} p_\lambda^2\, d\rho$, and weight each sample by $w(f) = N / \sum_\lambda p_\lambda(f)^2$.
5. **Fit.** Solve the weighted least-squares problem. It splits into one $N$-dimensional problem per output mode, all sharing one Gram matrix $\mathbf G$.

With $M \gtrsim N \log N$ optimally drawn samples, $\mathbf G$ is well conditioned with high probability, no matter how many output modes are used. The fitted operator is then close to the best approximation in the space.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/John-Turnage/OWLS_OpLearn.git
cd OWLS_OpLearn
pip install -r requirements.txt   # numpy, scipy, matplotlib, plus what the experiments need
pip install -e .                  # makes `import owls` work from anywhere
```

## Quickstart

```python
import numpy as np
import owls

d = 6
measures = [owls.Jacobi(j, j) for j in range(1, d + 1)]   # f_j ~ Jacobi(j, j) on [-1, 1]
index_set = owls.index_set(d, 4, kind="lp", p=1)          # polynomials of total degree <= 4

def solver(f):          # (M, d) input coefficients -> (M, d_out) output coefficients
    ...                 # e.g. run your PDE solver on each row

model = owls.learn_operator(solver, measures, index_set, rng=0)   # samples, solves, fits
print(model.cond())                                               # condition number of G (< 3 w.h.p.)

f_test = owls.sample_rho(1000, measures, rng=1)                   # test inputs f ~ rho
print(owls.relative_bochner_error(model.predict(f_test), solver(f_test)))
```

`python examples/quickstart.py` runs a complete version of this example on a nonlinear problem in about 20 seconds. It compares optimal sampling with plain Monte Carlo.

## Applying it to your own problem

1. **Encode the inputs.** Choose an orthonormal basis for your input functions and keep $d$ coefficients. For example, use the leading modes of a spectral basis, chosen by energy (`owls.energy_truncation`), or PCA coefficients of a dataset.
2. **Choose the input measure.** Pick a list of 1-D measures, one per coefficient: `owls.Jacobi(alpha, beta)` on $[-1, 1]$, `owls.Gaussian(mean, std)`, or `owls.Gamma(shape, scale)`. A quick way to set the decay of the coefficients is `owls.Jacobi(alpha)` with `alpha = owls.alpha_from_variance(var)`.
3. **Write a solver** that maps an `(M, d)` array of input coefficients to an `(M, d_out)` array of output coefficients in an orthonormal output basis. Outputs may be complex.
4. **Choose an index set.** Use `owls.index_set(d, k, kind="hc" or "lp", weights=..., max_degree=...)`, or `owls.linear_index_set(n, d)` for linear operators. Smaller weights $w_j$ allow lower degrees in direction $j$.
5. **Fit and check.** Call `owls.learn_operator(solver, measures, index_set)`. It uses `M = owls.sample_size(N)` samples by default. Check `model.cond()` or `model.gram_deviation()`, then estimate the error on Monte Carlo test inputs (`owls.sample_rho`, `owls.relative_bochner_error`).

If you already have a dataset and cannot choose the inputs, use `owls.EmpiricalWLS(x_pool, y_pool, index_set, measures).fit(M)`. It orthonormalizes the polynomials on the data (via a QR factorization) and samples optimally from the dataset.

## Repository layout

```
owls/                    the package
  index_sets.py          hyperbolic cross and lp index sets
  measures.py            1-D measures and their orthonormal polynomials, Monte Carlo and optimal sampling
  wls.py                 BlockWLS (inputs you can sample), EmpiricalWLS (fixed datasets), learn_operator
  utils.py               sample_size, energy_truncation, error norms
  plotting.py            the figure style used in the paper
  _vendor/               orthogonal polynomial routines from UncertainSCI (MIT license)
examples/quickstart.py   a complete small example
experiments/             notebooks for the paper's numerical results (see experiments/README.md)
  poisson/  burgers/  navier_stokes/
ERRATA.md                bugs fixed since arXiv v1, and places where the paper text differs from the code
```

## Reproducing the paper

Each notebook in `experiments/` has two flags at the top:

- With `RECOMPUTE = False`, the notebook redraws the figure from the saved results in `data/` in a few seconds.
- With `RECOMPUTE = True`, it reruns the experiment.
- With `QUICK = True`, it runs a tiny version for testing, written to a temporary folder.

See [`experiments/README.md`](experiments/README.md) for the list of notebooks, which figure each one produces, and run times. The Navier–Stokes data must be downloaded separately (see `experiments/navier_stokes/raw_data/README.md`).

The code exactly as it was run for arXiv v1 is preserved at the git tag `arxiv-v1`. **One result changed after the cleanup:** the Poisson comparison of optimal and Monte Carlo sampling. The v1 code trained the Monte Carlo runs on too few samples; with the correct number, both are stable and equally accurate for this linear problem. [`ERRATA.md`](ERRATA.md) lists this and all other differences.

## Citation

```bibtex
@article{turnage2025owls,
  title   = {An Optimal Weighted Least-Squares Method for Operator Learning},
  author  = {Turnage, John and Lowery, Matthew and Jakeman, John and Morrow, Zachary and Narayan, Akil and Shankar, Varun},
  journal = {arXiv preprint arXiv:2512.11168},
  year    = {2025}
}
```

## Acknowledgments

The orthogonal polynomial routines in `owls/_vendor/` come from [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI) (MIT license). The Navier–Stokes dataset comes from Kossaifi et al., *Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs* (2023), [Zenodo 12825163](https://zenodo.org/records/12825163).

This work was supported by the Sandia National Laboratories Laboratory Directed Research and Development (LDRD) program. J. Turnage and A. Narayan were partially supported by AFOSR under grant FA9550-23-1-0749.

## License

MIT; see [`LICENSE`](LICENSE).
