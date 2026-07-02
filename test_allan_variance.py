"""TODO:
simulate effects of quantization, other effecs to see differnt slopes
simulate a clock and its noise model and plot its allan variance
show how 1/f flicker noise leads to divergence of "traditional" estimators
"""

import numpy as np
from numpy import sqrt, pi
from numpy import log as ln
from matplotlib import pyplot as plt
from FlickerNoise import FlickerNoiseGenerator
from AllanVariance import AVAR
from helper import plot_noise_lines, predict_adev
from time import time

np.random.seed(0)  # For determinism

# =============================== MODIFY ME: ===============================
t_max = 3600 * 100  # [s] how much time to generate noise for
f_sampling = 1  # [Hz] sample rate

angle_random_walk = 0.15  # [deg/sqrt(hr)]
bias_instability_deg_per_hr = 0.3  # [deg/hr]
rate_random_walk = 0.5  # [(deg/hr)/sqrt(hr)]

# =========================Simulation =========================
dt = 1.0 / f_sampling  # [s]
N_SAMPLES = int(t_max // dt)

sigma_white = np.deg2rad(angle_random_walk / 60)  # [(rad/s)/sqrt(Hz)]
bias_instability = np.deg2rad(bias_instability_deg_per_hr / 3600.0)  # [rad/s]
sigma_walk = np.deg2rad(rate_random_walk / (3600 * 60))  # [(rad/s^2)/sqrt(Hz)]

min_allan_var_without_flicker_noise = 2 * sigma_white * sigma_walk / sqrt(3)
assert min_allan_var_without_flicker_noise < bias_instability

additive_white_measurement_noise = np.random.normal(0.0, sigma_white / np.sqrt(dt), N_SAMPLES)
sigma_flicker = sqrt((bias_instability**2 - min_allan_var_without_flicker_noise) / (2 * ln(2) / pi))

flicker_noise, _ = FlickerNoiseGenerator(f_sampling, t_max, sigma_flicker).generate_noise()
bias_derivative_white_noise = np.random.normal(0.0, sigma_walk / np.sqrt(dt), N_SAMPLES)
random_walk = np.cumsum(bias_derivative_white_noise * dt)
gyro_rate_data = random_walk + flicker_noise + additive_white_measurement_noise


avar_object = AVAR(gyro_rate_data, dt)
START = time()
print("Starting AVAR computation ... ")
numTaus = 100
# confidence_method = "IEEE 952"
confidence_method = "howe"
avar_object.compute(numTaus)
print(time() - START)
# h = hashlib.sha1()
# h.update(adevs.tobytes())
# print(h.hexdigest())

taus = avar_object.taus
adevs = np.sqrt(avar_object.avars)

sigma_white_predicted, sigma_flicker_predicted, sigma_walk_predicted = avar_object.infer_noise_params()
adev_lower_bound, adev_upper_bound = avar_object.compute_error_bounds("standard chi", 0.95)

predicted_adev = predict_adev(taus, sigma_white_predicted, sigma_flicker_predicted, sigma_walk_predicted)
bias_instability_predicted = min(predicted_adev)
angle_random_walk = 60 * np.rad2deg(sigma_white_predicted)  # [deg/sqrt(hr)]
bias_instability = 3600 * np.rad2deg(bias_instability_predicted)  # [deg/hr]
rate_random_walk = 3600 * 60 * np.rad2deg(sigma_walk_predicted)  # [(deg/hr)/sqrt(hr)]

print(f"{sigma_white = } # [(rad/s)/sqrt(Hz)]")
print(f"{sigma_flicker = } # [rad/s]")
print(f"{sigma_walk = } # [(rad/s)/sqrt(Hz)]")
print()

print(f"{sigma_white_predicted = } # [(rad/s)/sqrt(Hz)]")
print(f"{sigma_flicker_predicted = } # [rad/s]")
print(f"{sigma_walk_predicted = } # [(rad/s^2)/sqrt(Hz)]")

print()
print(f"{angle_random_walk = } # [deg/sqrt(hr)]")
print(f"{bias_instability = } # [deg/hr]")
print(f"{rate_random_walk = } # [(deg/hr)/sqrt(hr)]")

units = "rad/s"
plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.plot(taus, adevs, label="Measurements (95% confidence)", linewidth=0.75)
plt.gca().fill_between(taus, adev_lower_bound, adev_upper_bound, alpha=0.3)
plot_noise_lines(plt.gca(), taus, sigma_white_predicted, sigma_flicker_predicted, sigma_walk_predicted)
plt.plot(taus, predicted_adev, marker=".", label="prediction")
plt.xlabel(r"Averaging time $\tau$ [s]")
plt.ylabel(r"$\sigma$($\tau$)" + f"\n[{units}]", rotation=0)
plt.title("Synthetic Gyroscope Data")
plt.legend()
plt.grid()
plt.gca().set_aspect("equal")
plt.tight_layout()

# ============================================ Plotting ============================================
# [rad/s] --> [deg/hr]
adevs = np.rad2deg(adevs * 3600)
predicted_adev = np.rad2deg(predicted_adev * 3600)
adev_lower_bound = np.rad2deg(adev_lower_bound * 3600)
adev_upper_bound = np.rad2deg(adev_upper_bound * 3600)
units = "deg/hr"

plt.figure()
plt.xscale("log")
plt.yscale("log")
plt.plot(taus, adevs, label="Measurements (95% confidence)", linewidth=0.75)
plt.gca().fill_between(taus, adev_lower_bound, adev_upper_bound, alpha=0.3)
plot_noise_lines(
    plt.gca(),
    taus,
    3600 * np.rad2deg(sigma_white_predicted),
    3600 * np.rad2deg(sigma_flicker_predicted),
    3600 * np.rad2deg(sigma_walk_predicted),
)
plt.plot(taus, predicted_adev, marker=".", label="prediction")
plt.xlabel(r"Averaging time $\tau$ [s]")
plt.ylabel(r"$\sigma$($\tau$)" + f"\n[{units}]", rotation=0)
plt.title("Synthetic Gyroscope Data")
plt.legend()
plt.grid()
plt.gca().set_aspect("equal")
plt.tight_layout()

plt.show()
