"""
Helpful Resources:
https://www.phidgets.com/docs/Allan_Deviation_Primer
https://telesens.co/wp-content/uploads/2017/05/AllanVariance5087-1.pdf
https://en.wikipedia.org/wiki/Allan_variance
"""

import numpy as np
from numpy import pi, sqrt, log10
from numpy import log as ln
from helper import chi_squared_confidence_interval, predict_avar
import scipy
from multiprocessing import shared_memory, Pool


def avar_helper(x, dt, m, show_progress=True):
    """
    dt: sampling period [s]
    m: number of samples to average
    """
    A = x[2 * m :]
    B = x[m:-m]
    C = x[: -2 * m]

    A2BC = A - 2 * B + C
    tmp = np.mean(A2BC * A2BC)
    tau = m * dt
    allan_variance = tmp / (2 * (m**2))
    if show_progress:
        print(f"Finished tau = {tau} seconds")
    return tau, allan_variance


def avar_helper_shared_memory(shm, shape, dt, m, show_progress=True):
    x = np.ndarray(shape, buffer=shm.buf)
    return avar_helper(x, dt, m, show_progress)


class AVAR:
    def __init__(self, y=None, dt=None):
        self.y = y  # the measured signal
        if dt is not None:
            self.dt = float(dt)
        self.N = None
        self.taus = None
        self.avars = None

        self.sigma_white = None
        self.sigma_flicker = None
        self.sigma_walk = None

        self.adev_lower_bound = None
        self.adev_upper_bound = None

    def compute(self, numTaus, show_progress=True):
        """Special case of the general M-sample variance estimator with M=2 values of the average data value"""

        """
        numTaus: how many averaging times for which to calcualte the M-Sample Variance
        """
        self.N = len(self.y)  # Number of datapoints
        M = 2  # 2 samples for allan variance
        avg_window_sizes = np.geomspace(1.0, (self.N / M) - 1.0, num=numTaus, endpoint=True)
        # at this point the index interval lengths are floating point (bad)

        avg_window_sizes = np.ceil(avg_window_sizes).astype(int)
        # now the index interval lengths are integers but there could've been duplicates

        self.avg_window_sizes = np.unique(avg_window_sizes)
        # now the index interval lengths are all different integers

        x = np.cumsum([0, *self.y])  # integral of y
        self.taus = []
        self.avars = []

        enough_data_to_use_multiprocessing = len(x) > 1e3
        if enough_data_to_use_multiprocessing:
            # Create shared memory block
            shm = shared_memory.SharedMemory(create=True, size=x.nbytes)
            shared_array = np.ndarray(x.shape, buffer=shm.buf)
            shared_array[:] = x  # Copy data into shared SharedMemory

            # Compute avar at different tau values in parallel
            with Pool() as pool:
                results = pool.starmap(
                    avar_helper_shared_memory,
                    [(shm, x.shape, self.dt, m, show_progress) for m in self.avg_window_sizes],
                )

            # Clean up shared memory
            shm.close()
            shm.unlink()
            for i, m in enumerate(self.avg_window_sizes):
                tau, avar = results[i]
                self.taus.append(tau)
                self.avars.append(avar)
        else:
            for m in self.avg_window_sizes:
                tau, avar = avar_helper(x, self.dt, m, show_progress)
                self.taus.append(tau)
                self.avars.append(avar)

        self.taus = np.array(self.taus)
        self.avars = np.array(self.avars)

    def number_of_inliers(self, predicted_avar):
        # find number of data points that fit in the error bars
        within_error_bounds = np.logical_and(
            (self.avar_lower_bound < predicted_avar), (predicted_avar < self.avar_upper_bound)
        )
        return sum(within_error_bounds)

    def get_loss(self, predicted_avars):
        truth = log10(self.avars)
        prediction = log10(predicted_avars)
        error = np.abs(truth - prediction)
        return sum(error)

    def get_loss2(self, log10_params):
        return self.get_loss(predict_avar(self.taus, *(10**log10_params)))

    def infer_noise_params(self, infer_white=True, infer_flicker=True, infer_walk=True):
        """Guess a bunch of values and see which ones give the best fit"""

        # Step 1/2: Find possible ranges for the parameters
        # ======================================================================================
        possible_sigma_white = np.sqrt(self.taus * self.avars)
        possible_sigma_flicker = np.sqrt(self.avars / (2 * ln(2) / pi))
        possible_sigma_walk = np.sqrt(3 * self.avars * np.reciprocal(self.taus))

        # divide by 100 to give some margin on the lower end
        min_sigma_white = possible_sigma_white.min() / 100
        min_sigma_flicker = possible_sigma_flicker.min() / 100
        min_sigma_walk = possible_sigma_walk.min() / 100

        # Don't need margin on the upper end since the allan variance is positive (margin on the upper end can only lead to a worse fit)
        max_sigma_white = possible_sigma_white.max()
        max_sigma_flicker = possible_sigma_flicker.max()
        max_sigma_walk = possible_sigma_walk.max()

        # Step 2/2: Grid-search the parameter space on a log scale with a 0.2 decade discretization
        # ======================================================================================
        lowest_error = 1e99
        best_parameters = (0, 0, 0)
        discretization = 0.2  # [decades]
        N_white = int(np.ceil(np.log10(max_sigma_white / min_sigma_white) / discretization))
        N_flicker = int(np.ceil(np.log10(max_sigma_flicker / min_sigma_flicker) / discretization))
        N_walk = int(np.ceil(np.log10(max_sigma_walk / min_sigma_walk) / discretization))
        for sigma_white in [0, *np.geomspace(min_sigma_white, max_sigma_white, N_white, endpoint=True)]:
            for sigma_flicker in [0, *np.geomspace(min_sigma_flicker, max_sigma_flicker, N_flicker, endpoint=True)]:
                for sigma_walk in [0, *np.geomspace(min_sigma_walk, max_sigma_walk, N_walk, endpoint=True)]:
                    sigma_white = sigma_white if infer_white else 0.0
                    sigma_flicker = sigma_flicker if infer_flicker else 0.0
                    sigma_walk = sigma_walk if infer_walk else 0.0

                    predicted_avar = predict_avar(self.taus, sigma_white, sigma_flicker, sigma_walk)
                    error = self.get_loss(predicted_avar)
                    if error < lowest_error:
                        lowest_error = error
                        best_parameters = (sigma_white, sigma_flicker, sigma_walk)
        optimized_params = self.optimize_noise_params(*best_parameters)
        self.sigma_white, self.sigma_flicker, self.sigma_walk = optimized_params
        return optimized_params

    def optimize_noise_params(self, sigma_white, sigma_flicker, sigma_walk, bounds=[(0, 1e99), (0, 1e99), (0, 1e99)]):
        (
            (min_sigma_white, max_sigma_white),
            (min_sigma_flicker, max_sigma_flicker),
            (min_sigma_walk, max_sigma_walk),
        ) = bounds
        initial_params = np.array(
            [
                log10(sigma_white) if sigma_white >= min_sigma_white else -1e99,
                log10(sigma_flicker) if sigma_flicker >= min_sigma_flicker else -1e99,
                log10(sigma_walk) if sigma_walk >= min_sigma_walk else -1e99,
            ]
        )

        res = scipy.optimize.minimize(
            self.get_loss2,
            initial_params,
            bounds=log10(
                [
                    (min_sigma_white, max_sigma_white) if sigma_white > 0 else (0, 0),
                    (min_sigma_flicker, max_sigma_flicker) if sigma_flicker > 0 else (0, 0),
                    (min_sigma_walk, max_sigma_walk) if sigma_walk > 0 else (0, 0),
                ]
            ),
        )
        optimized_params = 10**res.x
        return optimized_params

    def get_noise_params(self):
        assert (
            self.sigma_white is not None and self.sigma_flicker is not None and self.sigma_walk is not None
        ), "Noise parameters have not been computed yet; perhaps you forgot to call infer_noise_params()?"
        return self.sigma_white, self.sigma_flicker, self.sigma_walk

    def compute_error_bounds(self, confidence_method=None, confidence_level=None, using_optical_gyro=False):
        assert confidence_method is not None

        adev_lower_bound = []
        adev_upper_bound = []
        for i, (adev, m) in enumerate(zip(np.sqrt(self.avars), self.avg_window_sizes)):
            N = self.N  # alias for brevity
            if confidence_method == "IEEE 952":
                relative_error = 1 / sqrt((N / m - 1))
                if using_optical_gyro:  # Tehrani explained optical gyros have 2 beams, resulting in sqrt(2) factor
                    relative_error /= sqrt(2)
                lower = adev / (1 + relative_error)
                upper = adev / (1 - relative_error)
            elif confidence_method == "standard chi":
                assert confidence_level is not None, "You forgot to specify a confidence level"
                dofs = N / m - 1
                lower, upper = chi_squared_confidence_interval(adev, dofs, confidence_level)
            else:
                assert confidence_level is not None, "You forgot to specify a confidence level"
                white_FM_dofs = ((3 * (N - 1) / (2 * m)) - (2 * (N - 2) / N)) * (4 * (m**2) / (4 * (m**2) + 5))
                flicker_FM_dofs = 2 * ((N - 2) ** 2) / (2.3 * N - 4.9) if m == 1 else 5 * N * N / (4 * m * (N + 3 * m))
                random_walk_FM_dofs = ((N - 2) / m) * (((N - 1) ** 2) - 3 * m * (N - 1) + 4 * (m**2)) / ((N - 3) ** 2)
                # print(white_FM_dofs, flicker_FM_dofs, random_walk_FM_dofs)

                assert (
                    self.sigma_white is not None and self.sigma_flicker is not None and self.sigma_walk is not None
                ), "Noise parameters have not been computed yet; perhaps you forgot to call infer_noise_params()?"
                white_noise_contribution = predict_avar(self.taus, self.sigma_white, 0, 0)
                flicker_noise_contribution = predict_avar(self.taus, 0, self.sigma_flicker, 0)
                random_walk_noise_contribution = predict_avar(self.taus, 0, 0, self.sigma_walk)

                error_white_noise = np.abs(self.avars - white_noise_contribution)
                error_flicker_noise = np.abs(self.avars - flicker_noise_contribution)
                error_random_walk_noise = np.abs(self.avars - random_walk_noise_contribution)

                use_white_noise_bound = error_white_noise < np.min(
                    [error_flicker_noise, error_random_walk_noise], axis=0
                )
                use_flicker_noise_bound = error_flicker_noise < np.min(
                    [error_white_noise, error_random_walk_noise], axis=0
                )
                use_random_walk_noise_bound = error_random_walk_noise < np.min(
                    [error_white_noise, error_flicker_noise], axis=0
                )

                # ensure mutually exclusive
                assert np.all(
                    np.sum([use_white_noise_bound, use_flicker_noise_bound, use_random_walk_noise_bound], axis=0) == 1
                )

                if use_white_noise_bound[i]:
                    lower, upper = chi_squared_confidence_interval(adev, white_FM_dofs, confidence_level)
                elif use_flicker_noise_bound[i]:
                    lower, upper = chi_squared_confidence_interval(adev, flicker_FM_dofs, confidence_level)
                elif use_random_walk_noise_bound[i]:
                    lower, upper = chi_squared_confidence_interval(adev, random_walk_FM_dofs, confidence_level)
                else:
                    assert False, "Shouldn't have reached here"

            adev_lower_bound.append(lower)
            adev_upper_bound.append(upper)
        return np.array(adev_lower_bound), np.array(adev_upper_bound)

    def cache_allan_variance_results_in_filesystem(self, results_cache_filepath):
        with open(results_cache_filepath, "w") as f:
            f.write(f"{self.N}\n")
            f.write(f"{self.dt}\n")
            np.savetxt(f, np.array([self.avg_window_sizes, self.taus, self.avars]).T)

    def read_cached_allan_variance_results_from_filesystem(self, results_cache_filepath):
        with open(results_cache_filepath, "r") as f:
            self.N = float(f.readline())  # Read the first line as number
            self.dt = float(f.readline())  # Read the first line as number
            self.avg_window_sizes, self.taus, self.avars = np.loadtxt(f).T
