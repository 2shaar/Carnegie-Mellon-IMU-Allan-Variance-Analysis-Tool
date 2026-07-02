import numpy as np
from numpy import sqrt
from time import time

def sampleVariance(data):
    "data: list of datapoints whose sample (not population) variance is to be found"
    return np.var(np.array(data), ddof=1)  # setting ddof=1 conveys to numpy to use len(data)-1 degrees of freedom


def M_SampleVariance_helper(data, dt, avg_window_size, M, T_idx):
    """
    data: dataset
    M: number of samples to estimate the variance of the average data value over an interval of length avg_window_size indicies
    T_idx: duration between the beginnings of successive averaging intervals (measured in array indicies instead of time)
    avg_window_size: duration of the interval (measured in array indicies instead of time) to compute the average.
    Can't index into x based on a (floating point) time; can only do so by an (integer) index.
    So, the x values to be accessed are avg_window_size apart.


    returns:
    the average of N - (M * avg_window_size) datapoints of
    the variance of M values of
    the average data value within an interval spanned by avg_window_size indices
    with each of the N - (M * avg_window_size) intervals starting at indices 0,1,..,N - (M * avg_window_size)-1.

    Note: the M intervals used to calculate average data value overlap and are only shifted by 1 index
    """
    START = time()
    N = len(data)
    assert 0 < avg_window_size
    assert 0 < N - M * avg_window_size + 1

    """
    The accumulated values act as a lookup table to make the M-Sample Variance faster when it needs to compute the average 
    data value over an interval.
    """

    variances = []
    averaging_kernel = np.ones(avg_window_size) / avg_window_size
    averaged = np.convolve(data, averaging_kernel, "valid")
    for n in range(0, N - M * avg_window_size + 1):  # TODO speed up this loop
        averages = averaged[n::T_idx][:M]
        variances.append(sampleVariance(averages))
    new_answer = np.average(variances)
    print(f"New: {time()-START:.2f} s")

    return new_answer


def M_SampleVariance(data, dt, M, numTaus, T_idx):
    """
    T_idx: duration between the beginnings of successive averaging intervals (measured in array indicies instead of time)
    numTaus: how many averagign times for which to calcualte the M-Sample Variance
    """
    N = len(data)  # Number of datapoints

    avg_window_sizes = np.geomspace(1.0, (N / M) - 1.0, num=numTaus, endpoint=True)
    # at this point the index interval lengths are floating point (bad)

    avg_window_sizes = np.ceil(avg_window_sizes).astype(int)
    # now the index interval lengths are integers but there could've been duplicates

    avg_window_sizes = np.unique(avg_window_sizes)
    # now the index interval lengths are all different integers
    var = []
    dev = []
    taus = []
    for avg_window_size in avg_window_sizes:
        tau = dt * avg_window_size
        taus.append(tau)
        sample_variance = M_SampleVariance_helper(data, dt, avg_window_size, M, T_idx)
        var.append(sample_variance)

        """
        Technically the square root of the sample variance is a BIASED estimator of the sample standard deviation, so the Allan deviation should not be interpreted as a legitimate measure of sample standard deviation. Since the units of Allan Variance are SU (Signal Units) squared, taking the square root yields a more natural quantity with the same units as the data collected. Thus, Allan deviation should not be interpreted as an independent quantity but rather as "the square root of Allan Variance so the units make more sense"
        """
        dev.append(sqrt(sample_variance))

        print("Finished tau = {} seconds".format(tau))
    return taus, var, dev
