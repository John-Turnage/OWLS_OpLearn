"""
Helpers for the Navier-Stokes experiments: loading the dataset, encoding the inputs
(PCA) and outputs (Fourier modes orthonormal in H^alpha), and plotting.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import ticker


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
def load_ns_data(raw_dir="raw_data"):
    """
    Load the forcing -> vorticity dataset (see raw_data/README.md for the download).

    Returns X_train, Y_train, X_test, Y_test as numpy arrays of shape (S, 128, 128).
    """
    import torch

    paths = [os.path.join(raw_dir, f"nsforcing_{s}_128.pt") for s in ("train", "test")]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. See raw_data/README.md for download instructions.")
    train = torch.load(paths[0], weights_only=True)
    test = torch.load(paths[1], weights_only=True)
    return train["x"].numpy(), train["y"].numpy(), test["x"].numpy(), test["y"].numpy()


def encode_inputs(X_train_grid, X_test_grid, energy=0.995, n_max=750, normalization="uniform", random_state=0):
    """
    PCA encoding of the input (forcing) fields, scaled to [-1, 1].

    Keeps the smallest number of PCA modes d_in capturing `energy` of the training
    variance (at most n_max). The coefficients are then divided by a common constant
    ("uniform", keeps relative sizes) or column by column ("entrywise") so that they
    lie in [-1, 1], the domain of the Jacobi polynomials. As in the paper's runs, the
    scaling constant is the largest |coefficient| over train *and* test data.

    random_state seeds sklearn's randomized SVD. (The paper's runs used no seed, which
    can change d_in-dependent quantities like N_eff by one.)

    Returns x_train (S, d_in), x_test, and the fitted sklearn PCA object.
    """
    from sklearn.decomposition import PCA

    X_train = X_train_grid.reshape(X_train_grid.shape[0], -1)
    X_test = X_test_grid.reshape(X_test_grid.shape[0], -1)

    pca = PCA(n_components=min(n_max, *X_train.shape), random_state=random_state).fit(X_train)
    d_in = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), energy) + 1

    pca = PCA(n_components=d_in, random_state=random_state)
    x_train = pca.fit_transform(X_train)
    x_test = pca.transform(X_test)

    if normalization == "uniform":
        linf = np.max(np.abs(np.vstack((x_train, x_test))))
        x_train, x_test = x_train / linf, x_test / linf
    elif normalization == "entrywise":
        linf = np.max(np.abs(np.vstack((x_train, x_test))), axis=0)
        x_train, x_test = x_train / linf, x_test / linf
    else:
        raise ValueError("normalization must be 'uniform' or 'entrywise'")
    return x_train, x_test, pca


def encode_outputs(Y_train_grid, Y_test_grid, alpha, energy=0.999):
    """
    Encode the output (vorticity) fields by their H^alpha-orthonormal Fourier coefficients,
    keeping the modes that carry `energy` of the average H^alpha energy of the training data.

    Returns y_train (S, d_out), y_test (complex arrays), and the fitted HAlphaBasisCompressor.
    """
    compressor = HAlphaBasisCompressor(N=Y_train_grid.shape[1], threshold=energy, alpha=alpha)
    compressor.fit(Y_train_grid)
    return compressor.project(Y_train_grid), compressor.project(Y_test_grid), compressor


# ---------------------------------------------------------------------------------------
# Output basis
# ---------------------------------------------------------------------------------------
class HAlphaBasisCompressor:
    """
    Fourier basis on the periodic square [0, L)^2 that is orthonormal in H^alpha:

        psi_k(x) = exp(i k.x) / (L (1 + |k|^2)^(alpha/2)),

    with k the (physical) wavenumber. The coefficient of g on psi_k is
    (1 + |k|^2)^(alpha/2) times its L^2-orthonormal Fourier coefficient, so the
    Euclidean norm of the coefficient vector is the H^alpha norm of g.
    (alpha = 0 gives L^2, alpha = 1 gives H^1, negative alpha gives weaker norms.)

    fit() keeps the modes with the largest average H^alpha energy over training data,
    until a fraction `threshold` of the total is reached.
    """

    def __init__(self, N, L=2 * np.pi, threshold=0.99, alpha=1):
        """
        N : grid points per axis (grid is N x N on [0, L)^2)
        L : period
        threshold : fraction of the average H^alpha energy to keep
        alpha : Sobolev exponent
        """
        self.N = N
        self.L = L
        self.threshold = threshold
        self.alpha = alpha
        self.basis_indices = None  # flat indices of the retained modes
        self.M = None              # number of retained modes (= d_out)
        self.weights = self._compute_weights(N)

    def _compute_weights(self, N):
        """H^alpha weights (1 + kx^2 + ky^2)^(alpha/2) on the FFT grid of wavenumbers."""
        kx = 2 * np.pi * np.fft.fftfreq(N, d=self.L / N)
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        return np.power(1 + KX**2 + KY**2, self.alpha / 2)

    def _coefficients(self, dat):
        # L^2-orthonormal Fourier coefficients (FFT scaled by L / N^2), times the H^alpha weights
        return np.fft.fft2(dat) * (self.L / self.N**2) * self.weights

    def fit(self, Y_grid):
        """Choose the retained modes from training data Y_grid of shape (S, N, N)."""
        avg_energy = np.zeros(self.N * self.N)
        for dat in Y_grid:
            avg_energy += np.abs(self._coefficients(dat).flatten()) ** 2
        avg_energy /= Y_grid.shape[0]

        order = np.argsort(avg_energy)[::-1]
        cumulative = np.cumsum(avg_energy[order])
        fraction = cumulative / cumulative[-1]
        self.M = np.searchsorted(fraction, self.threshold) + 1
        self.basis_indices = order[:self.M]
        return self

    def project(self, dat):
        """Reduced H^alpha coefficients of one field (N, N) -> (M,), or of a stack (S, N, N) -> (S, M)."""
        if self.basis_indices is None:
            raise ValueError("The compressor must be fit before projecting data.")
        coeffs = self._coefficients(dat)
        return coeffs.reshape(coeffs.shape[:-2] + (-1,))[..., self.basis_indices]

    def reconstruct_coeffs(self, reduced_coeffs):
        """Full (N, N) array of H^alpha coefficients, zero outside the retained modes."""
        full = np.zeros(self.N * self.N, dtype=complex)
        full[self.basis_indices] = reduced_coeffs
        return full.reshape(self.N, self.N)

    def reconstruct_spatial(self, reduced_coeffs):
        """Field on the N x N grid from its reduced H^alpha coefficients."""
        l2_coeffs = self.reconstruct_coeffs(reduced_coeffs) / self.weights
        return np.fft.ifft2(l2_coeffs).real * (self.N**2 / self.L)


# ---------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------
class OperatorLearningPlotter:
    """Side-by-side plots of true vs. predicted vorticity, and error histograms."""

    def __init__(self, domain=(0, 2 * np.pi, 0, 2 * np.pi), alpha=0, color_map="viridis", err_color_map="inferno"):
        self.domain = domain
        self.alpha = alpha
        self.color_map = color_map
        self.err_color_map = err_color_map

    def plot_comparison(self, true_field, approx_field, spectral_error=None, rel_spectral_error=None,
                        title_prefix="", save_as=None):
        """Plot omega, omega_tilde and |omega - omega_tilde|; errors (in H^alpha) go in the title."""
        abs_err = np.abs(true_field - approx_field)
        normalizer = Normalize(np.min([true_field, approx_field]), np.max([true_field, approx_field]))
        err_normalizer = Normalize(np.min(abs_err), np.max(abs_err))

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3, figure=fig)
        panels = [(true_field, r"$\omega$", self.color_map, normalizer),
                  (approx_field, r"$\tilde{\omega}$", self.color_map, normalizer),
                  (abs_err, r"$\vert \omega - \tilde{\omega}\vert$", self.err_color_map, err_normalizer)]
        for i, (field, title, cmap, norm) in enumerate(panels):
            ax = plt.subplot(gs[0, i])
            ax.imshow(field, extent=self.domain, origin="lower", cmap=cmap, norm=norm)
            ax.set_title(title)
            ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

        title = title_prefix
        norm_label = r"\Vert_{H^{%s}}" % self.alpha
        if spectral_error is not None:
            title += "\n" + r"$\Vert \omega - \tilde{\omega}" + norm_label + "$ = " + f"{spectral_error:.2E}"
        if rel_spectral_error is not None:
            title += "     " + r"$\mathrm{Rel}\Vert \omega - \tilde{\omega}" + norm_label + "$ = " + f"{rel_spectral_error:.2E}"
        plt.suptitle(title, y=0.85)

        fig.colorbar(cm.ScalarMappable(norm=normalizer, cmap=self.color_map),
                     cax=fig.add_axes([0.125, 0.29, 0.502, 0.02]), orientation="horizontal")
        cb = fig.colorbar(cm.ScalarMappable(norm=err_normalizer, cmap=self.err_color_map),
                          cax=fig.add_axes([0.672, 0.29, 0.228, 0.02]), orientation="horizontal")
        cb.locator = ticker.MaxNLocator(nbins=3)
        cb.update_ticks()
        if save_as is not None:
            plt.savefig(save_as, bbox_inches="tight")
        plt.show()

    def plot_error_distribution(self, errors, title_prefix="", bins=None, save_as=None):
        """Histogram of per-sample errors with the mean marked."""
        if bins is None:
            bins = max(10, round(len(errors) / 10))
        mean_err = np.mean(errors)
        plt.figure()
        plt.hist(errors, bins=bins, density=True, color="black")
        plt.axvline(mean_err, color="red", linestyle="dotted", linewidth=2, label=f"Mean: {mean_err:.2e}")
        plt.title(r"{t} $H^{{{a}}}$ Error Distribution".format(t=title_prefix, a=self.alpha))
        plt.xlabel("Error")
        plt.ylabel("Density")
        plt.legend()
        if save_as is not None:
            plt.savefig(save_as, bbox_inches="tight")
        plt.show()
