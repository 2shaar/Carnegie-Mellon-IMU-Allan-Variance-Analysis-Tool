from tqdm import tqdm
import numpy as np
from numpy import pi, sqrt
from multiprocessing import Pool


class FlickerFilterStage:
    def __init__(self, stage, f_sampling):
        Ts = 1.0 / f_sampling  # sampling period [s]

        # Backwards Euler Discretization: s = (1-(1/z))/T_s
        self.a = (1 + Ts * (3 ** (2 * stage))) / (3 + Ts * (3 ** (2 * stage)))
        self.b = -1 / (3 + Ts * (3 ** (2 * stage)))
        self.c = 3 / (3 + Ts * (3 ** (2 * stage)))
        print(stage, self.a, self.b, self.c)
        self.prev_input = 0.0
        self.prev_output = 0.0

    def update(self, current_input):
        current_output = self.a * current_input + self.b * self.prev_input + self.c * self.prev_output
        self.prev_output = current_output
        self.prev_input = current_input
        return current_output


class FlickerFilterStageStateSpace:
    def __init__(self, stage, f_sampling):
        Ts = 1.0 / f_sampling  # sampling period [s]

        # Backwards Euler Discretization: s = (1-(1/z))/T_s

        # modeling this as a discrete, linear dynamical system in state-space
        a = (1.0 / Ts) + 3 ** (2 * stage)
        b = -1.0 / Ts
        c = (3.0 / Ts) + 3 ** (2 * stage)
        d = -3.0 / Ts
        self.x = np.zeros((2, 1))  # [y[n], u[n]]
        self.A = np.array(
            [
                [-d / c, b / c],
                [0.0, 0.0],
            ]
        )
        self.B = np.array([[a / c], [1.0]])
        self.C = np.array([[1.0, 0.0]])

    def update(self, u):
        self.x = self.A @ self.x + self.B * u
        y = self.C @ self.x
        return y.item()


class FlickerNoiseGenerator:
    def __init__(self, sampling_frequency, t_max, sigma_flicker, seed=None):
        # For generating noise with PSD = sigma_flicker^2 / w

        self.fs = sampling_frequency
        self.t_max = t_max
        self.sigma_flicker = sigma_flicker

        if seed is not None:
            np.random.seed(seed)

        w_l = 2 * pi / t_max  # low frequency limit (could be lowered more, but # of stages and complexity increases)
        w_u = 2 * pi * self.fs  # upper frequency limit

        # stage number for the filter cascade at the lowest end of the signal bandwidth
        # solving 3^(2*lowest_stage-1) = f_low
        lowest_stage = 0.5 * (1 + np.log(w_l) / np.log(3))

        # stage number for the filter cascade at the highest end of the signal bandwidth
        # solving 3^(2*highest_stage) = f_high
        highest_stage = 0.5 * (np.log(w_u) / np.log(3))

        lowest_stage = np.floor(lowest_stage).astype(int)
        highest_stage = np.ceil(highest_stage).astype(int)
        assert lowest_stage < highest_stage

        self.filter_stages = [FlickerFilterStageStateSpace(i, self.fs) for i in range(lowest_stage, highest_stage + 1)]
        self.N_stages = len(self.filter_stages)
        self.x = np.zeros((2 * self.N_stages, 1))
        self.A = np.zeros((2 * self.N_stages, 2 * self.N_stages))
        self.B = np.zeros((2 * self.N_stages, 1))
        self.C = np.zeros((1, 2 * self.N_stages))

        for k in range(self.N_stages):
            self.A[2 * k : 2 * (k + 1), 2 * k : 2 * (k + 1)] = self.filter_stages[k].A
            if k == 0:
                self.B[0:2] = self.filter_stages[0].B
            else:
                self.A[2 * k : 2 * (k + 1), 2 * (k - 1) : 2 * k] = self.filter_stages[k].B @ self.filter_stages[k].C
            if k == self.N_stages - 1:
                self.C[0, -2] = 1.0

        # Want PSD(w) = sigma_flicker^2 / w
        # Equivalently, |H(w)| = sigma_flicker/sqrt(w)
        # But, the filter cascade's |H(w)| will be something else: |H(w)| = gamma / sqrt(w) before scaling
        # So, need to find the necesssary scaling for the flicker noise to have the desired sigma_flicker

        # ---------- Finding the scaling ----------
        # Since the filter cascade only has 1/sqrt(w) falloff within the bandwidth [w_low, w_high],
        # we should evaluate it at the 'middle' of the frequency decades, i.e. geometric mean
        w_mid = np.sqrt(w_l * w_u)  # geometric mean
        H_mag = 1
        for stage in range(lowest_stage, highest_stage + 1):
            H_numerator = 1j * (w_mid) + 3 ** (2 * stage)
            H_denominator = 3j * (w_mid) + 3 ** (2 * stage)
            H_mag *= np.abs(H_numerator / H_denominator)
        gamma = H_mag * sqrt(w_mid)
        self.scaling = self.sigma_flicker / gamma

    def generate_noise(self, compute_cov=False):
        dt = 1.0 / self.fs
        N = int(self.t_max // dt)
        sigma_input = 1.0 / sqrt(dt)  # white noise with PSD of 1
        unscaled_noise = []
        output_var = []
        self.x_cov = np.zeros((2 * self.N_stages, 2 * self.N_stages))
        for i in tqdm(range(N), desc="Simulating noise... "):
            u = np.random.normal(0, sigma_input)
            self.x = self.A @ self.x + self.B * u
            output = self.C @ self.x
            unscaled_noise.append(output.item())

            if compute_cov:
                self.x_cov = self.A @ self.x_cov @ self.A.T + self.B @ self.B.T * sigma_input**2
                output_cov = self.C @ self.x_cov @ self.C.T
                output_var.append(output_cov.item())
        return self.scaling * np.array(unscaled_noise), np.sqrt(output_var) * self.scaling


def get_noise(f_sampling, t_max, sigma_flicker, seed):
    return FlickerNoiseGenerator(f_sampling, t_max, sigma_flicker, seed).generate_noise()


def show_ensemble(f_sampling, t_max, sigma_flicker):

    N = 10
    results = []
    with Pool() as pool:
        results = pool.starmap(get_noise, [(f_sampling, t_max, sigma_flicker, i) for i in np.arange(N)])
    t = np.arange(len(results[0][0])) / f_sampling
    for flicker_noise, output_sigma in results:
        t = np.arange(len(flicker_noise)) / f_sampling
        plt.plot(t, flicker_noise, linewidth=0.1)
    plt.grid()
    plt.plot(t, 3 * output_sigma, color="red", linestyle="--")
    plt.plot(t, -3 * output_sigma, color="red", linestyle="--")
    plt.xlabel("t [hr]")
    plt.ylabel(r"[${}^\circ$/hr]")
    plt.show()


def show_PSD(f_sampling, t_max, sigma_flicker):
    noise, _ = FlickerNoiseGenerator(f_sampling, t_max, sigma_flicker, 0).generate_noise()
    N_SAMPLES = len(noise)
    f, S_x = signal.welch(noise, f_sampling, nperseg=2 ** int(np.log2(N_SAMPLES) - 6), return_onesided=True)
    plt.figure()
    plt.loglog(f, S_x, linewidth=0.2, label="measured")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD")
    plt.gca().set_aspect("equal")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from scipy import signal
    from numpy import log as ln

    np.set_printoptions(linewidth=999, precision=3)
    np.random.seed(0)

    f_sampling = 3600  # [samples/hr]
    t_max = 10  # [hr]
    bias_instability = 0.4
    sigma_flicker = bias_instability / sqrt(2 * ln(2) / pi)

    from scipy.linalg import solve_discrete_lyapunov

    fng = FlickerNoiseGenerator(f_sampling, t_max, sigma_flicker, 0)
    sigma_input = sqrt(f_sampling)
    cov_ss = solve_discrete_lyapunov(fng.A, fng.B @ fng.B.T * sigma_input**2)
    assert 0 < min(np.linalg.eigvals(cov_ss))
    print(sqrt(fng.C @ cov_ss @ fng.C.T) * fng.scaling)
    print(sqrt(ln(f_sampling * t_max)) * fng.scaling)
    # exit(1)

    show_ensemble(f_sampling, t_max, sigma_flicker)
    # show_PSD(f_sampling, t_max, sigma_flicker)
