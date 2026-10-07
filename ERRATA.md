# Errata and differences from arXiv v1

The code exactly as it was run for the first arXiv version (arXiv:2512.11168v1) is preserved at the git tag `arxiv-v1`. This file lists:

1. bugs that were fixed afterwards, and which results changed;
2. places where the v1 paper text describes a setting differently from what the code actually ran. The code was kept as run, so that the saved data and figures stay valid.

Equation, section and figure numbers below refer to arXiv v1.

## 1. Bugs fixed (results regenerated)

**Poisson, conditioning and accuracy (Figure 3, Section 5.1).**
In the Monte Carlo ("standard sampling") runs, the training inputs were drawn with the number of *test* samples (500) instead of $M = c_\delta N_{\mathrm{eff}} \log(4 N_{\mathrm{eff}})$.

- At $N_{\mathrm{eff}} = 500$ the least-squares system was therefore square, which caused the spike in $\mathrm{cond}(\mathbf G)$.
- For $N_{\mathrm{eff}} > 500$ it was underdetermined. The reported condition numbers then fell again, and the test error grew.

With the correct $M$, Monte Carlo and optimal sampling give essentially the same condition numbers (about 1.7–1.9) and errors for this linear problem. Figure 3 was regenerated, and the discussion of the spike near $N_{\mathrm{eff}} = 500$ and of Monte Carlo over-fitting no longer applies. (For linear operators with these near-Gaussian input marginals, Monte Carlo sampling is already close to optimal. Optimal sampling matters for the polynomial spaces of the Burgers and Navier–Stokes examples.)

**Poisson, learned operator (Figure 4a).**
The plotted matrix was the coefficient matrix $C$ with respect to $\Phi_{\mathbf n} = \sigma_{\mathbf n}^{-1}\psi \otimes \xi_{\mathbf n}$, labelled $\langle \tilde K_V \xi_{\mathbf n}, \xi_{\mathbf m}\rangle$. The two differ by a factor $1/\sigma_{\mathbf n}$ per row. The figure now shows $\langle \tilde K_V \xi_{\mathbf n}, \xi_{\mathbf m}\rangle$. The diagonal structure is unchanged, but the color scale changes.

**Poisson, Green's function (Figure 5).**
- The "relative $L^2$ kernel error" was the *squared* relative error, while the Bochner error next to it was not squared.
- The learned kernel used only the diagonal of the learned operator, dropping the (small) off-diagonal terms.

Both are fixed. The kernel error is now $\|G - \tilde G\|_{L^2}/\|G\|_{L^2}$ against the exact kernel, including the modes beyond 208 that the learned kernel cannot represent. Figure 5 was regenerated.

The corrected kernel error lies *above* the Bochner error, for example $5.5\times 10^{-4}$ vs. $6.8\times 10^{-6}$ at $N_{\mathrm{eff}} = 100$. The kernel norm weights all input modes equally, while the Bochner norm weights them by their variance $1/(2n^2+3)$. Almost all of the kernel error comes from the input modes $n > N_{\mathrm{eff}}$ that the learned operator ignores: $\big(90 \sum_{n > 100} (n\pi)^{-4}\big)^{1/2} \approx 5.5 \times 10^{-4}$. Also, the Figure 5 caption says $M(208)$ training samples; the code uses $M(N_{\mathrm{eff}})$ for each $N_{\mathrm{eff}}$.

**Burgers, output energy (Figure 8a).**
The expectations were averaged over 100 samples from the optimal measure $\mu$ (with no weights), while the caption says $f \sim \rho$. The data are now computed from 100 Monte Carlo samples $f \sim \rho$. Figure 8a was regenerated.

**Smaller fixes (no effect on the paper's results):**
- The induced (optimal) sampler could index one past the end of its table when a uniform draw exceeded the last CDF value.
- Weights were left undefined if `induced` was not exactly `True`/`False`.
- The `index_set` docstring gave the wrong formula for the $\ell^p$ set.
- The Poisson Green's function notebook saved two files to `data/` instead of `Data/` (breaks on case-sensitive file systems).
- No random seeds were set; all notebooks now take a `SEED`.

**Speed.** The optimal sampler rebuilt a 1000-point Gauss rule for every input coordinate and looped over samples in Python. It now caches the rules (Poisson has 1225 coordinates but only 69 distinct measures) and is vectorized. For the same random stream it produces bit-for-bit the same samples as the v1 code.

## 2. Paper text vs. code (code kept as run)

**Index sets (eqs. (5.1), (5.2)).**
In the code the anisotropy weights $w_j \in (0,1]$ enter as an exponent $1/w_j$:

- hyperbolic cross: $\prod_j (1 + \lambda_j)^{1/w_j} \le k + 1$, i.e. $\|\gamma \odot \log(\lambda + 1)\|_1 \le \log(k+1)$ with $\gamma_j = 1/w_j$;
- $\ell^p$ set: $\big(\sum_j \lambda_j^{p/w_j}\big)^{1/p} \le k$. This is not $\|\gamma \odot \lambda\|_p \le k$.

A smaller $w_j$ allows lower degrees in direction $j$. With the paper's $\gamma_j$ entering as $\gamma \odot \lambda$, the decreasing weights would instead *favor* the high modes.

**Viscous Burgers (Section 5.2).**
- Input measure: the code uses $\alpha_n = (n-1)^2$ for $n = 1, \dots, 20$ (`[a**2 for a in range(20)]`), not $\alpha_n = n^2$. The first mode is therefore uniform on $[-1, 1]$.
- Input truncation: the 20 input modes carry 96.8% of the input energy under the $\alpha_n$ actually used, and 94.9% under $\alpha_n = n^2$, rather than 95%.
- Anisotropy weights: `linspace(1, 0.01, 20)`, i.e. $\Delta = 0.99/19$ (the paper says $0.99/20$).
- Solver resolution: the Galerkin solve used $d_{\mathrm{solve}} = 200$ modes for Figures 6b and 8b, 150 for Figure 7, and 500 for Figure 8a, with time steps $10^{-4}$ (Figures 6b, 8a, and 8b for $\nu = 0.001$) or $10^{-3}$ (Figure 7, and 8b for $\nu \ge 0.01$). The paper says $d_{\mathrm{solve}} > 10\max\{d_{\mathrm{in}}, d_{\mathrm{out}}\}$.
- Figure 6a: the largest index set, HC(60), was capped at 5000 of its 5046 indices.
- Figure 8b: the marker shows the first $d_{\mathrm{out}}$ whose error is within 1% of the error at $d_{\mathrm{out}} = 192$.

**Navier–Stokes (Section 5.3).**
- The PCA keeps 99.5% of the forcing variance (the paper says 95.5%); this gives $d_{\mathrm{in}} = 142$.
- The inputs are scaled to $[-1,1]$ by the largest coefficient over the training *and* test data.
- The reference Jacobi measures match the variance of each PCA coefficient; the mean is zero after PCA.
- Figure 9a omits the largest index set ($k = 24$).

**Poisson (Section 5.1).** The 1-D Green's function example keeps 208 modes, which carry 99.495% of the input energy (rounded to 99.5% in the paper).

## 3. Reproducibility notes

- The v1 runs used no random seeds. The data in `experiments/*/data/` from those runs are kept as they were, and rerunning gives statistically similar but not identical numbers.
- The Navier–Stokes PCA used scikit-learn's randomized SVD without a seed. As a result $N_{\mathrm{eff}}$ in `Ns.npy` is 3547 or 3548 for the same settings. `encode_inputs` now takes a `random_state`.
- Figures 7 and 10 (Burgers example solutions, Navier–Stokes median sample) were produced in v1 without saving their data. Their notebooks now save it when rerun.
- `experiments/navier_stokes/figures/NS_med_err.png` comes from a v1 run with $\alpha = -4$ (its title shows $H^{-4}$ and a relative error of $4.6\times 10^{-2}$). Figure 10 and the notebook use $\alpha = -2$. Rerunning the notebook with $\alpha = -2$ gives a relative $H^{-2}$ error of about 0.13 for the median-error test sample (0.105 in the paper) and a median of 0.12 over the test set.
