# Vendored code from UncertainSCI

The files in this folder come from [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI)
(Copyright (c) 2020 The Scientific Computing and Imaging Institute, MIT license; see `LICENSE`).
They provide the orthogonal polynomial families (Jacobi, Hermite, Laguerre): three-term
recurrences, evaluation and Gauss quadrature. `owls.measures` uses them.

| File | Upstream file |
|---|---|
| `families.py` | `UncertainSCI/families.py` |
| `opoly1d.py` | `UncertainSCI/opoly1d.py` |
| `transformations.py` | `UncertainSCI/transformations.py` |
| `casting.py` | `UncertainSCI/utils/casting.py` |

These copies were taken from UncertainSCI around late 2020 / early 2021 (closest upstream
commits: `50caf7d` for `families.py`, `bed195b` for `opoly1d.py`), with local modifications:

- `JacobiPolynomials` has the methods `eval_1d` and `eval_nd`.
- Imports are relative (`from .opoly1d import ...`).
- `HermitePolynomials` and `LaguerrePolynomials`: the probability normalization `ab[0, 1] = 1`
  is now also applied when only degree 0 is requested (`N >= 0` instead of `N > 0`). Before,
  evaluating p_0 on its own returned the wrong constant.
