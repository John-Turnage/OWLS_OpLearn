# Experiments

Each folder has one notebook per result, a `data/` folder with the saved results, and a
`figures/` folder with the figures. Every notebook starts with two flags:

```python
RECOMPUTE = False  # True: rerun the experiment and overwrite data/
QUICK = False      # True: tiny version of the experiment, written to a temporary folder (for testing)
```

- With the defaults, a notebook redraws its figure from `data/` in a few seconds.
- Notebooks that take only about a minute have no `RECOMPUTE` flag and always run in full.

Install the package first (`pip install -e .` in the repository root). The figures use LaTeX
fonts if `latex` and `dvipng` are installed, and matplotlib's Computer Modern fonts otherwise.

## Notebooks

"Status" says whether the saved data come from the arXiv v1 run (code at the git tag `arxiv-v1`),
or were regenerated with the cleaned-up code after a bug fix (see `../ERRATA.md`).

| Notebook | What it computes | Figure in arXiv v1 | Status | Run time |
|---|---|---|---|---|
| `poisson/conditioning_and_error` | cond(G) and test error vs. N_eff, optimal vs. Monte Carlo sampling | Fig. 3 (`cond_err.pdf`) | regenerated | 10 min |
| `poisson/learn_2d_operator` | learned operator matrix; median-error test sample | Fig. 4 (`approx_operator.pdf`, `median_err.pdf`) | regenerated | 40 s |
| `poisson/greens_function_1d` | learned Green's function vs. the exact one | Fig. 5 (`rel_l2_kernel_and_boch_err.pdf`, `kernel_pointwise_err.pdf`) | regenerated | 30 s |
| `burgers/conditioning` | cond(G) vs. N_eff (no PDE solves) | Fig. 6a (`vB_CondNum.pdf`) | v1 | 7 min |
| `burgers/error_vs_index_set` | test error vs. index set and viscosity | Fig. 6b (`vB_Err.pdf`) | v1 | ≈ 3 h (est.) |
| `burgers/example_solutions` | example predictions; test error distribution | Fig. 7 (`vB_err_dist.pdf`) | v1 (data not saved) | ≈ 5 min (est.) |
| `burgers/output_energy` | output energy lost by truncating to d_out modes | Fig. 8a (`vb_out_truc_err.pdf`) | regenerated | 45 s |
| `burgers/output_dimension` | test error vs. d_out | Fig. 8b (`vb_truncation_error_with_viscosity.pdf`) | v1 | ≈ 30 min (est.) |
| `navier_stokes/conditioning` | cond(G) vs. N_eff, optimal vs. uniform sampling of the dataset | Fig. 9a (`NS_CondNum.pdf`) | v1 | 2.5 min |
| `navier_stokes/error_vs_sobolev_alpha` | test error and dim Y_h vs. the output norm H^alpha | Fig. 9b (`NS_err.pdf`) | v1 | 3 min |
| `navier_stokes/learn_operator` | median-error test sample, alpha = -2 | Fig. 10 (`NS_med_err.png`) | regenerated | 1 min |

Run times are for a full rerun (`RECOMPUTE = True`) on an Apple M3 Pro (12 cores, 18 GB RAM). The largest
Poisson case needs about 5 GB of memory. "est." marks run times estimated from the measured speed of the Burgers solver (about 46 solves per second at 200 modes and 2000 time steps) rather than timed. The Navier–Stokes notebooks need the dataset in
`navier_stokes/raw_data/` (see the README there) and up to about 7 GB of memory (the $\alpha = 5$ case of
`error_vs_sobolev_alpha` has 14231 complex output coefficients).

## Saved data

| File | Contents |
|---|---|
| `poisson/data/Dims.npy` | the N_eff values |
| `poisson/data/Condition_{Ind,Std}.npy` | cond(G), optimal (`Ind`) and Monte Carlo (`Std`) sampling; shape (N_eff values, runs) |
| `poisson/data/Err_{Ind,Std}.npy` | empirical Bochner test error, same layout |
| `poisson/data/ker_N_eff.npy`, `ker_rel_boch_err.npy`, `ker_rel_kern_err.npy` | 1-D Green's function: N_eff, relative Bochner error, relative L^2 kernel error |
| `poisson/data/ker_Zdiffs.npy` | pointwise kernel errors on a 500 x 500 grid, for N_eff = 1, 10, 100 |
| `burgers/data/Condition_{Dims,Ind,Std}.npy` | N_eff and cond(G) for HC(10), ..., HC(60) |
| `burgers/data/{hc,lp}_{dims,errs}.npy` | N_eff and relative test errors (viscosity x index set) |
| `burgers/data/output_energy.npy` | mean squared output coefficients, viscosity x mode |
| `burgers/data/out_dim_dat.npy` | relative test error, viscosity x d_out = 2, 12, ..., 192 |
| `navier_stokes/data/Condition_{Dims,Ind,Std}.npy` | N_eff and cond(G) for HC(6), ..., HC(24) |
| `navier_stokes/data/{alphas,rel_Ha_errs,Y_dims,Ns}.npy` | alpha, mean relative H^alpha test error, dim Y_h, N_eff |
| `navier_stokes/data/learn_operator.npz` | alpha = -2: per-sample test errors, and true and predicted vorticity of the median-error test sample |
