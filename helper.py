import numpy as np
import random
import scipy
from numpy import sqrt, log10, pi
from numpy import log as ln
from matplotlib import pyplot as plt

random.seed(0)  # for determinism


def chi_squared_confidence_interval(sigma, dof, confidence=0.95):
    # from ChatGPT
    """
    Computes the confidence interval for the standard deviation (sigma)
    using the Chi-Squared distribution.

    Parameters:
        sigma (float): Sample standard deviation.
        dof (int): Degrees of freedom
        confidence (float): Confidence level (default is 95%).

    Returns:
        tuple: (lower bound, upper bound) of confidence interval for sigma.
    """
    alpha = 1 - confidence  # Confidence level
    chi2_lower = scipy.stats.chi2.ppf(alpha / 2, dof)  # Lower critical value
    chi2_upper = scipy.stats.chi2.ppf(1 - alpha / 2, dof)  # Upper critical value

    # Confidence interval for variance
    var_lower = (dof * sigma**2) / chi2_upper
    var_upper = (dof * sigma**2) / chi2_lower

    # Confidence interval for standard deviation
    return np.sqrt(var_lower), np.sqrt(var_upper)


def predict_avar(taus, sigma_white, sigma_flicker, sigma_walk):
    return (
        ((sigma_white**2) * np.reciprocal(taus)) + ((sigma_flicker**2) * 2 * ln(2) / pi) + (taus * (sigma_walk**2) / 3)
    )


def predict_adev(taus, sigma_white, sigma_flicker, sigma_walk):
    return sqrt(predict_avar(taus, sigma_white, sigma_flicker, sigma_walk))


def get_bias_instability(taus, sigma_white, sigma_flicker, sigma_walk):
    return min(predict_adev(taus, sigma_white, sigma_flicker, sigma_walk))


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
