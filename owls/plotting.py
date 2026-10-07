"""
Matplotlib settings used for the figures in the paper.
"""
import os
import shutil
import warnings

import matplotlib as mpl

TEXTWIDTH_PT = 422.52252  # LaTeX \textwidth of the paper, in points


def _latex_available():
    env = os.environ.get("OWLS_USETEX")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "")
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def paper_style(width=1.0, aspect=0.6, fontsize=7, legend_fontsize=None, usetex=None):
    """
    Set matplotlib rcParams to match the paper and return a figure size.

    Parameters
    ----------
    width : float
        Figure width as a fraction of the paper's text width.
    aspect : float
        Height / width.
    fontsize : float
        Base font size (tick labels; axis labels and titles are one point larger).
    legend_fontsize : float, optional
        Defaults to fontsize.
    usetex : bool, optional
        Use LaTeX for text. Default: only if `latex` and `dvipng` are installed
        (set the environment variable OWLS_USETEX=0 or 1 to override). Without
        LaTeX, matplotlib's built-in Computer Modern math fonts are used instead.

    Returns
    -------
    (fig_width, fig_height) in inches.
    """
    if usetex is None:
        usetex = _latex_available()
        if not usetex and "OWLS_USETEX" not in os.environ:
            warnings.warn("LaTeX not found: using matplotlib's Computer Modern math fonts instead.")

    mpl.rcParams.update({
        "savefig.bbox": "tight",
        "savefig.transparent": True,
        "savefig.format": "pdf",
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": fontsize,
        "axes.labelsize": fontsize + 1,
        "axes.titlesize": fontsize + 1,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize if legend_fontsize is None else legend_fontsize,
    })
    if usetex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "Computer Modern",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })
    else:
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
        })

    fig_width = TEXTWIDTH_PT / 72.27 * width
    return fig_width, fig_width * aspect


def savefig(path, fig=None):
    """Save a figure with tight bounding box. PDFs are written without a creation date, so re-saving gives a stable file."""
    import matplotlib.pyplot as plt

    kwargs = {"bbox_inches": "tight"}
    if str(path).endswith(".pdf"):
        kwargs["metadata"] = {"CreationDate": None}
    (fig if fig is not None else plt).savefig(path, **kwargs)
