import os

import numpy as np
from matplotlib import pyplot as plt


def save_figure_deterministic(figure, *args, **kwargs):
    # Deterministic Figure saving
    filename = args[0]
    metadata = kwargs.pop("metadata", {})
    if filename.endswith(".svg"):
        metadata.setdefault("Date", None)
    elif filename.endswith(".pdf"):
        metadata.setdefault("CreationDate", None)
    kwargs["metadata"] = metadata
    figure.savefig(*args, **kwargs)


def save_figure(
    figure,
    folder,
    filename_without_extension,
    close_after_saving=True,
    dpi="figure",
    figure_size=(10, 6),  # fits a google slide's aspect ratio
):
    assert " " not in folder
    assert " " not in filename_without_extension

    os.makedirs(folder, exist_ok=True)

    figure_has_content = any(ax.has_data() for ax in plt.gcf().axes)
    if not figure_has_content:
        print(f"Empty figure detected. Not saving figure for '{filename_without_extension}'")
        return

    figure.set_size_inches(*figure_size)
    path_wo_ext = os.path.join(folder, filename_without_extension)
    pdf_path = f"{path_wo_ext}.pdf"
    save_figure_deterministic(figure, pdf_path, bbox_inches="tight", dpi=dpi)

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
