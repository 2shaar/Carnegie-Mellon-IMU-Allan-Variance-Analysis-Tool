import numpy as np
from matplotlib import pyplot as plt

from AllanVariance import AVAR
from FlickerNoise import FlickerNoiseGenerator
from helper import predict_adev

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"


np.random.seed(0)  # For determinism


results_cache_filepath = "example_IMU_data/EpsonG364/Allan_Variance_Analysis_Results_1hr/.Gyroscope_Z_cached_allan.txt"

# ============================ Get Noise model from real data ============================
avar_real = AVAR()
avar_real.read_cached_allan_variance_results_from_filesystem(results_cache_filepath)
sigma_white, sigma_flicker, sigma_walk = avar_real.infer_noise_params()

# ========================= Simulate Noise Using the Noise Model =========================
dt = avar_real.dt  # [s] alias for brevity
f_sampling = 1 / dt  # [Hz]
t_max = avar_real.N * dt  # [s] how much time to generate noise for
N_SAMPLES = int(t_max // dt)

white_noise = np.random.normal(0.0, sigma_white / np.sqrt(dt), N_SAMPLES)
flicker_noise, _ = FlickerNoiseGenerator(f_sampling, t_max, sigma_flicker).generate_noise()
bias_derivative_white_noise = np.random.normal(0.0, sigma_walk / np.sqrt(dt), N_SAMPLES)
random_walk = np.cumsum(bias_derivative_white_noise * dt)
simulated_noise = white_noise + flicker_noise + random_walk

# ============================ Compute AVAR for Simulated Noise ============================
avar_sim = AVAR(simulated_noise, dt)
numTaus = 100
confidence_method = "howe"
avar_sim.compute(numTaus)

# ================ Compare ADEV for Real Noise, Model, and Simulated Noise ================
adev_real = np.sqrt(avar_real.avars)
adev_model = predict_adev(avar_real.taus, sigma_white, sigma_flicker, sigma_walk)
adev_sim = np.sqrt(avar_sim.avars)

# [rad/s] --> [deg/hr]
adev_real = np.rad2deg(adev_real * 3600)
adev_model = np.rad2deg(adev_model * 3600)
adev_sim = np.rad2deg(adev_sim * 3600)

fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(avar_real.taus, adev_real, label="Real Noise", linewidth=0.75)
ax.plot(avar_real.taus, adev_model, label="Noise Model")
ax.plot(avar_real.taus, adev_sim, label="Simulated Noise")
ax.set_xlabel(r"Averaging time $\tau$ [s]")
ax.set_ylabel(r"$\sigma$($\tau$)" + f"\n[deg/hr]", rotation=0)
ax.set_title("Comparing Allan Deviation for Real Noise, Noise Model, and Simulated Noise")
ax.legend()
ax.grid()
ax.set_aspect("equal")
ax.tick_params(axis="both", which="minor", labelleft=False, labelbottom=False)
plt.tight_layout()
plt.savefig("real_vs_model_vs_sim.pdf")
plt.show()
