import numpy as np
import scipy
from numpy import log as ln
from numpy import pi, sqrt


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
