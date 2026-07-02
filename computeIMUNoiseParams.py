# example usage:  python3 computeIMUNoiseParams.py FILENAME.csv
# GYRO DATA MUST BE IN rad/sec
# ACCEL DATA MUST BE IN m/s^2

import os
import sys
from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from pandas import read_csv
from plot_utils import hist, save_figure

from AllanVariance import AVAR
from helper import plot_noise_lines, predict_adev

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

USE_CACHE = 0  # TODO Remove me


#####################################################################################################
# Reading data file passed in to cmd line
received_args = len(sys.argv) - 1
if received_args != 1:
    print("Expected exactly 1 argument (file path) but received {received_args}")
    exit(1)

imu_csv_filepath = os.path.realpath(sys.argv[1])
imu_csv_filename = os.path.basename(imu_csv_filepath)
imu_csv_file_dirpath = os.path.dirname(imu_csv_filepath)

results_dirpath = os.path.join(
    imu_csv_file_dirpath,
    "Allan_Variance_Analysis_Results_"
    + os.path.splitext(imu_csv_filename.replace(" ", "_"))[0],  # strip file extension
)
os.system(f"mkdir -p '{results_dirpath}'")


class Logger(object):
    # https://stackoverflow.com/a/14906787
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


#####################################################################################################
if not USE_CACHE:
    print(f"Loading data from {imu_csv_filepath}")

    my_data = read_csv(imu_csv_filepath, sep=",").values
    t = my_data[:, 0]
    t -= t[0]
    data_dict = {
        "Gyroscope": [my_data[:, 1], my_data[:, 2], my_data[:, 3]],
        "Accelerometer": [my_data[:, 4], my_data[:, 5], my_data[:, 6]],
    }

    print("Done loading data")

    # visualizing the distribution of dts between samples for real imu data
    plt.figure(figsize=(8, 6), dpi=300)
    dts = np.diff(t)

    dt = np.mean(dts)

    hist(dts * 1000, bin_width=1.0)
    plt.xlabel("dt [ms]")
    plt.savefig(os.path.join(results_dirpath, "dt_histogram.png"))
    plt.close()
    plt.figure(figsize=(8, 6), dpi=300)
    dts = np.diff(t)
    hist(dts * 1000, bin_width=1.0, yscale="log")
    plt.xlabel("dt [ms]")
    plt.savefig(os.path.join(results_dirpath, "dt_histogram_log.png"))
    plt.close()

# copying file descriptor for output file to stdout so results that get printed to terminal are also saved in output results file
sys.stdout = Logger(os.path.join(results_dirpath, "output.txt"))

sensors = ["Gyroscope", "Accelerometer"]
fig, axes = plt.subplots(4, 2, figsize=(8, 8), dpi=300, sharex=True, sharey="col")
for i in range(2):
    sensor = sensors[i]
    if not USE_CACHE:
        histogram = plt.figure(figsize=(8, 6), dpi=300)
        log_histogram = plt.figure(figsize=(8, 6), dpi=300)

    for axis_number, axis in enumerate("XYZ"):
        series_label = f"{sensor}_{axis}"
        ####################################################################################################

        results_cache_filepath = os.path.join(results_dirpath, f".{series_label}_cached_allan.csv")  # hidden file
        if not USE_CACHE:
            series = data_dict[sensor][axis_number]
            assert not np.any(np.isnan(series))
            numTaus = 100
            avar_object = AVAR(series, dt)
            avar_object.compute(numTaus)
            avar_object.cache_allan_variance_results_in_filesystem(results_cache_filepath)
        else:
            avar_object = AVAR()
            avar_object.read_cached_allan_variance_results_from_filesystem(results_cache_filepath)

        taus, adevs = avar_object.taus, np.sqrt(avar_object.avars)
        sigma_white, sigma_flicker, sigma_walk = avar_object.infer_noise_params()
        adev_lower_bound, adev_upper_bound = avar_object.compute_error_bounds("standard chi", 0.95)

        # fit a non-flicker model
        # avar_object2 = AVAR()
        # avar_object2.read_cached_allan_variance_results_from_filesystem(results_cache_filepath)
        # sigma_white, sigma_flicker, sigma_walk = avar_object2.infer_noise_params(infer_flicker=0)

        predicted_adev = predict_adev(taus, sigma_white, sigma_flicker, sigma_walk)
        bias_instability = min(predicted_adev)

        # print(sigma_white, sigma_flicker, bias_instability, sigma_walk)

        print("=" * 100)
        print(series_label)
        dt = avar_object.dt
        if sensor == "Gyroscope":
            print(f"Estimated Angle Random walk: {60*np.rad2deg(sigma_white):.3g} [deg/sqrt(hr)]")
            print(f"\t <==> {sigma_white:.3g} [(rad/s)/sqrt(Hz)]")
            print(f"\t <==> {np.rad2deg(sigma_white):.3g} [(deg/s)/sqrt(Hz)]")
            print(
                f"\t <==> {sigma_white/sqrt(dt):.3g} [rad/s] additive white noise to angular velocity at samples {dt:.3g}s apart"
            )

            print(f"Estimated sigma_flicker: {3600 *np.rad2deg(sigma_flicker):.3g} [deg/hr]")
            print(f"\t <==> {sigma_flicker:.3g} [rad/s]")

            print(f"Estimated Bias Instability: {3600 *np.rad2deg(bias_instability):.3g} [deg/hr]")
            print(f"\t <==> {bias_instability:.3g} [rad/s]")

            print(f"Estimated Angular Rate Random walk: {3600*60*np.rad2deg(sigma_walk):.3g} [(deg/hr) / sqrt(hr)]")
            print(f"\t <==> {sigma_walk:.3g} [(rad/s^2) / sqrt(Hz)]")
            print(
                f"\t <==> {sigma_walk/sqrt(dt):.3g} [rad/s^2] additive white noise to angular acceleration rate at samples {dt:.3g}s apart"
            )
        else:
            g = 9.81
            print(f"Estimated Velocity Random walk: {sigma_white*60:.3g} [(m/s) / sqrt(hr)]")
            print(f"\t <==> {sigma_white:.3g} [(m/s^2) / sqrt(Hz)]")
            print(f"\t <==> {sigma_white*1e3/g:.3g} [mg / sqrt(Hz)]")
            print(f"\t <==> {sigma_white*1e6/g:.3g} [ug / sqrt(Hz)]")
            print(
                f"\t <==> {sigma_white/sqrt(dt):.3g} [m/s^2] additive white noise to linear acceleration at samples {dt:.3g}s apart"
            )

            print(f"Estimated sigma_flicker: {sigma_flicker:.3g} [m/s^2]")
            print(f"\t <==> {sigma_flicker*1e3/g:.3g} [mg]")
            print(f"\t <==> {sigma_flicker*1e6/g:.3g} [ug]")

            print(f"Estimated Bias Instability: {bias_instability:.3g} [m/s^2]")
            print(f"\t <==> {bias_instability*1e3/g:.3g} [mg]")
            print(f"\t <==> {bias_instability*1e6/g:.3g} [ug]")

            print(f"Estimated Acceleration Random walk: {sigma_walk*60:.3g} [(m/s^2) / sqrt(hr)]")
            print(f"\t <==> {sigma_walk:.3g} [(m/s^3)/sqrt(Hz)]")
            print(f"\t <==> {(sigma_walk*1e3/g)*60:.3g} [mg / sqrt(hr)]")
            print(f"\t <==> {(sigma_walk*1e6/g)*60:.3g} [ug / sqrt(hr)]")
            print(
                f"\t <==> {sigma_walk/sqrt(dt):.3g} [m/s^3] additive white noise to linear jerk at samples {dt:.3g}s apart"
            )
        print("=" * 100)

        # ------------------------------------ plotting results ------------------------------------
        if sensor == "Gyroscope":  # converting units to deg/hr to make the plot more intuitive
            adevs = 3600 * np.rad2deg(adevs)
            adev_lower_bound = 3600 * np.rad2deg(adev_lower_bound)
            adev_upper_bound = 3600 * np.rad2deg(adev_upper_bound)
            sigma_white = 3600 * np.rad2deg(sigma_white)
            sigma_flicker = 3600 * np.rad2deg(sigma_flicker)
            bias_instability = 3600 * np.rad2deg(bias_instability)
            sigma_walk = 3600 * np.rad2deg(sigma_walk)
            predicted_adev = 3600 * np.rad2deg(predicted_adev)
            units = r"$^\circ/\mathrm{hr}$"
        else:
            units = r"$\mathrm{m/s^2}$"

        if not USE_CACHE:
            if sensor == "Gyroscope":  # converting units to deg/hr to make the plot more intuitive
                series = 3600 * np.rad2deg(series)
            # visualizing the distribution of the time series data
            plt.figure(histogram)
            plt.subplot(3, 1, axis_number + 1)
            hist(series, nbins=100, label=axis)

            plt.figure(log_histogram)
            plt.subplot(3, 1, axis_number + 1)
            hist(series, nbins=100, yscale="log", label=axis)

        plt.figure(fig)
        # individual axes
        ax = axes[axis_number][i]
        ax.tick_params(  # disable x ticks
            axis="x",
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,  # labels along the bottom edge are off
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(taus, adevs, label="Measurements (95% confidence)", linewidth=0.75)
        ax.fill_between(taus, adev_lower_bound, adev_upper_bound, alpha=0.3)
        ax.plot(taus, predicted_adev, label="Model Fit", linewidth=0.75)
        ax.set_ylabel(rf"$\sigma$($\tau$)   [{units}]", fontsize=6)
        plot_noise_lines(ax, taus, sigma_white, sigma_flicker, sigma_walk)

        # Overall legend at the top
        if axis_number == 0:
            if i == 0:
                ax.legend(bbox_to_anchor=(1.3, 1.15), loc="lower center", prop={"size": 6}, ncols=5, framealpha=1.0)
            ax.set_title(sensor.capitalize(), fontsize=7)

        current_miny, current_maxy = ax.get_ylim()
        for _ in range(4):
            axes[_][i].set_ylim(
                min([min(adev_lower_bound), min(predicted_adev), current_miny]),
                max([max(adev_upper_bound), max(predicted_adev), current_maxy]),
            )

        ax.grid()
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

        props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=plt.rcParams["legend.edgecolor"])
        ax.text(
            0.5,
            0.95,
            f"Axis: {axis}",
            transform=ax.transAxes,
            fontsize=5,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=props,
        )

        # all 3 axes in 1 plot. No model fitting
        axes[3][i].loglog(
            taus, adevs, label=axis, color="tab:" + (["red", "green", "blue"][axis_number]), linewidth=0.5
        )
        axes[3][i].set_ylabel(rf"$\sigma$($\tau$)   [{units}]", fontsize=7)
        axes[3][i].legend(
            title="All 3 axes",
            fontsize=5,
            title_fontsize=5,
            ncols=3,
            framealpha=1.0,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            facecolor="white",
        )
        axes[3][i].grid()
        axes[3][i].set_aspect("equal")
        axes[3][i].tick_params(labelsize=6)

    # each axis with fitted model in a subplot
    axes[3][i].set_xlabel(r"Averaging time $\tau$ [s]", fontsize=6)
    plt.subplots_adjust(hspace=0.03)  # Adjust spacing

    if not USE_CACHE:
        plt.figure(histogram)
        plt.xlabel(f"[{units}]")
        plt.subplot(3, 1, 1)
        plt.title(sensor.replace("_", " "))
        plt.savefig(os.path.join(results_dirpath, f"{sensor}_series_histogram.png"))
        plt.close()

        plt.figure(log_histogram)
        plt.xlabel(f"[{units}]")
        plt.subplot(3, 1, 1)
        plt.title(sensor.replace("_", " "))
        plt.savefig(os.path.join(results_dirpath, f"{sensor}_series_histogram_log.png"))
        plt.close()

plt.figure(fig)
# at this point, the left and right subplots may have differen heights. Need to compute how many decades are on the plot with the largest y range and then adjust bounds for the other column's subplots accordingly
max_decade_range = 1e-99
for row_of_axes in axes:
    for ax in row_of_axes:
        bottom_y_decade, top_y_decade = np.log10(ax.get_ylim())
        decade_range = top_y_decade - bottom_y_decade
        max_decade_range = max(max_decade_range, decade_range)
for row_of_axes in axes:
    for ax in row_of_axes:
        bottom_y_decade, top_y_decade = np.log10(ax.get_ylim())
        decade_range = top_y_decade - bottom_y_decade
        decade_padding = (max_decade_range - decade_range) / 2
        ax.set_ylim((10 ** (bottom_y_decade - decade_padding), 10 ** (top_y_decade + decade_padding)))


plt.suptitle(
    f"Allan Deviation (Computed using {int(round(avar_object.N*avar_object.dt/3600))} hours of data collected at {int(round(1/avar_object.dt))} Hz)",
    fontsize=8,
    # y=0.94,
    y=1,
)

save_figure(plt.gcf(), results_dirpath, "Allan_dev_model_fit", figure_size=(8, 4.5), bbox_inches="tight")
