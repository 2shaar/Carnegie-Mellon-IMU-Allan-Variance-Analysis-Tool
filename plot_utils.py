from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy import log as ln
from numpy import log10, pi


def save_figure_deterministic(figure, *args, **kwargs):
    # Deterministic Figure saving
    filename = args[0]
    metadata = kwargs.pop("metadata", {})
    if str(filename).endswith(".svg"):
        metadata.setdefault("Date", None)
    elif str(filename).endswith(".pdf"):
        metadata.setdefault("CreationDate", None)
    kwargs["metadata"] = metadata
    figure.savefig(*args, **kwargs)


def save_figure(
    figure,
    folder: Path,
    filename_without_extension,
    close_after_saving=True,
    dpi="figure",
    bbox_inches="tight",
    figure_size=(10, 6),  # fits a google slide's aspect ratio
    **kwargs,
):
    assert " " not in str(folder)
    assert " " not in str(filename_without_extension)

    folder.mkdir(parents=True, exist_ok=True)

    figure_has_content = any(ax.has_data() for ax in plt.gcf().axes)
    if not figure_has_content:
        print(f"Empty figure detected. Not saving figure for '{filename_without_extension}'")
        return

    figure.set_size_inches(*figure_size)
    pdf_path = folder / f"{filename_without_extension}.pdf"
    save_figure_deterministic(figure, pdf_path, bbox_inches=bbox_inches, dpi=dpi, **kwargs)

    if close_after_saving:
        plt.close(figure)


def hist(data, nbins=None, bin_width=None, yscale=None, label=None):
    min_value, max_value = min(data), max(data)

    # for getting percentage
    weights = 100 * np.ones_like(data) / float(len(data))

    if bin_width is not None:
        assert nbins is None
        nbins = int((max_value - min_value) / bin_width)
        if 30 < nbins < 1000:  # reasonable number of bins
            plt.hist(data, bins=np.linspace(min_value, max_value + bin_width, nbins), weights=weights, label=label)
        else:
            plt.hist(data, weights=weights, label=label)
    elif nbins is not None:
        bin_width = (max_value - min_value) / nbins
        plt.hist(data, bins=np.linspace(min_value, max_value + bin_width, nbins), weights=weights, label=label)
    else:
        plt.hist(data, weights=weights, label=label)

    if label is not None:
        plt.legend()
    plt.grid(True)
    if yscale is not None:
        plt.yscale("log")
    plt.ylabel("percentage of counts")


def plot_line_on_loglog(ax, taus, m, b, color, label=""):
    adjust_ylim = m == 0.0  # don't let drawing a line screw up the axes unless if it is a horizontal line

    ylim = ax.get_ylim()
    taus_plotted = []
    adevs = []
    for log10_tau in np.log10(taus):
        log10_adev = m * log10_tau + b
        adev = 10**log10_adev
        tau = 10**log10_tau
        taus_plotted.append(tau)
        adevs.append(adev)
    ax.plot(taus_plotted, adevs, label=label, linestyle="--", color=color, linewidth=0.5)
    if not adjust_ylim:
        ax.set_ylim(ylim)  # restore original y limits


def plot_noise_lines(ax, taus, sigma_white, sigma_flicker, sigma_walk):
    if sigma_white != 0.0:
        plot_line_on_loglog(ax, taus, -0.5, log10(sigma_white), "fuchsia", "Additive White Noise")
    if sigma_flicker != 0.0:
        plot_line_on_loglog(ax, taus, 0.0, 0.5 * log10((sigma_flicker**2) * 2 * ln(2) / pi), "grey", "Flicker Noise")
    if sigma_walk != 0.0:
        plot_line_on_loglog(ax, taus, 0.5, log10(sigma_walk) - 0.5 * np.log10(3), "lime", "Bias Random Walk")
