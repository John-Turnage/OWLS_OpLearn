import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import ticker
import numpy as np
from decimal import Decimal
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100

class OperatorLearningPlotter:
    def __init__(self, domain=(0, 2 * np.pi, 0, 2 * np.pi), alpha=0, color_map = 'viridis', err_color_map = 'inferno'):
        self.domain = domain
        self.alpha = alpha
        self.color_map = color_map
        self.err_color_map = err_color_map

    def plot_comparison(self, true_field, approx_field,  
                        spectral_error=None, spatial_error=None,
                        rel_spectral_error=None, rel_spatial_error=None,
                        title_prefix=""):
        abs_err = np.abs(true_field - approx_field)

        norm_min = np.min([true_field, approx_field])
        norm_max = np.max([true_field, approx_field])
        normalizer = Normalize(norm_min, norm_max)
        im = cm.ScalarMappable(norm=normalizer, cmap=self.color_map)

        err_normalizer = Normalize(np.min(abs_err), np.max(abs_err))
        err_im = cm.ScalarMappable(norm=err_normalizer, cmap=self.err_color_map)

        fig = plt.figure()
        gs = gridspec.GridSpec(1, 3, figure=fig)

        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(true_field, extent=self.domain, origin='lower', cmap=self.color_map, norm=normalizer)
        ax1.set_title(r'$\omega$ ')
        self._clean_ticks(ax1)

        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(approx_field, extent=self.domain, origin='lower', cmap=self.color_map, norm=normalizer)
        ax2.set_title(r'$\tilde{\omega}$')
        self._clean_ticks(ax2)

        ax3 = plt.subplot(gs[0, 2])
        ax3.imshow(abs_err, extent=self.domain, origin='lower', cmap=self.err_color_map, norm=err_normalizer)
        ax3.set_title(r'$\vert \omega - \tilde{\omega}\vert$')
        self._clean_ticks(ax3)

        title_lines = [title_prefix]
        if spectral_error is not None:
            spec_err_val = Decimal(spectral_error)
            title_lines.append("\nSpectral Error: ")
            title_lines.append(r"$\Vert c - \tilde{c}$" + r"$\Vert_{{H^{{{a}}}}}= $".format(a = self.alpha) + "{:.2E}".format(spec_err_val))
        if rel_spectral_error is not None:
            rel_spec_val = Decimal(rel_spectral_error)
            title_lines.append("     ")
            title_lines.append(r"$\text{Rel}\Vert c - \tilde{c}$" + r"$\Vert_{{H^{{{a}}}}}= $".format(a=self.alpha) + "{:.2E}".format(rel_spec_val))
        #if spatial_error is not None:
        #    spat_err_val = Decimal(spatial_error)
        #    title_lines.append("\nSpatial Error: ")
        ##    title_lines.append(r"$\Vert \omega - \tilde{\omega}$"+ r"$\Vert_{{H^{{{a}}}}}= $".format(a = self.alpha) + "{:.2E}".format(spat_err_val))
        #if rel_spatial_error is not None:
        #    rel_spat_val = Decimal(rel_spatial_error)
        #    title_lines.append("     ")
        #    title_lines.append(r"$\text{Rel}\Vert \omega - \tilde{\omega}$" + r"$\Vert_{{H^{{{a}}}}}= $".format(a = self.alpha) + "{:.2E}".format(rel_spat_val))

        plt.suptitle("".join(title_lines), y=0.85)

        cbar_ax = fig.add_axes([0.125, .29, 0.502, 0.02])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        err_cbar_ax = fig.add_axes([0.672, 0.29, 0.228, 0.02])
        cb = fig.colorbar(err_im, cax=err_cbar_ax, orientation='horizontal')
        cb.locator = ticker.MaxNLocator(nbins=3)
        cb.update_ticks()

        plt.show()

    def plot_error_distribution(self, errors, title_prefix="", bins=None):
        if bins is None:
            bins = max(10, round(len(errors) / 10))

        mean_err = np.mean(errors)

        plt.hist(errors, bins=bins, density=True, color='black')
        plt.axvline(mean_err, color='red', linestyle='dotted', linewidth=2, label=f"Mean: {mean_err:.2e}")
        plt.title(r"{t} $H^{{{a}}}$ Error Distribution".format(t=title_prefix, a= self.alpha)+"\nMean Error = {m:.2e}".format(m = mean_err))
        plt.xlabel(r"Error")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    def _clean_ticks(self, ax):
        ax.tick_params(top=False, bottom=False, left=False, right=False,
                       labelleft=False, labelbottom=False)
