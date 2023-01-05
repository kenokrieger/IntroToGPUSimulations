from glob import glob

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

try:
    plt.style.use("science")
except IOError:
    pass


POWER_LAW_REGION = (0.1, 2.0)  # x region of the power law
INFO = "\n".join((
    "Simulation Parameters:",
    "Size: 32$\\,$x$\\,$32",
    "$\\beta = 1.0$",
    "$j = 1.0$",
    "$\\alpha = 8.0$",
    "Iterations: $10^4$",
    "Spinflips per ns: 0.028"
))


def main():
    """
    Generate some statistics based on the recorded magnetisation of
    a simulation.
    """
    magnetisation = np.loadtxt("magnetisation.dat")
    plt.plot(magnetisation)
    plt.xlabel("Timesteps")
    plt.ylabel("Relative Magnetisation")
    plt.savefig("Magnetisation.pdf", dpi=300)
    plt.close("all")

    bins, log_returns, cumulative_returns = compute_statistics(magnetisation)
    plt.plot(bins, cumulative_returns, label="Cumulative returns")
    # select region for power law fitting
    power_law_bins = np.where((bins > POWER_LAW_REGION[0]) & (bins < POWER_LAW_REGION[1]))
    bins = bins[power_law_bins]
    cumulative_returns = cumulative_returns[power_law_bins]

    # fit the power law using scipy.optimize.curve_fit
    # generally, it is recommended to use MLE fitting when fitting to a histogram
    # to ensure that the weights of the measurements are considered
    # (there exist a lot more data points for lower values than for higher values)
    params = fit_power_law(bins, cumulative_returns)

    plt.plot(bins, 10 ** (fit_func(np.log10(bins), *params)),
             label=f"$a x^{{-\\tau}}$ with $\\tau={-params[0]:.2f}$",
             color="red")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1, 20_000)
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.legend(bbox_to_anchor=(1.68, 1.0))
    plt.text(1.05, 0.2, INFO, transform=plt.gca().transAxes)
    plt.savefig("Cumulative_Return_Distribution.pdf", dpi=300)
    return 0


def compute_statistics(magnetisation):
    """
    Given the recorded magnetisation of a simulation compute
    the return and cumulative return distribution.

    Args:
        magnetisation (np.ndarray): The recorded relative magnetisation of a system.

    Returns:
        None.

    """
    log_returns = np.abs(np.diff(np.log(np.abs(magnetisation))))
    start = np.log2(1e-3)
    end = np.log2(15)
    bins = np.logspace(start, end, base=2.0, num=500)
    counts, bins = np.histogram(np.abs(log_returns), bins=bins)

    non_zero_counts = np.where(counts > 0.0)
    bins = bins[non_zero_counts]
    counts = counts[non_zero_counts]

    np.savetxt(f"return_bins.dat", bins)
    np.savetxt(f"return_distribution.dat", counts)
    cumulative_distribution = np.flip(np.cumsum(np.flip(counts)))
    np.savetxt(f"cumulative_return_distribution.dat", cumulative_distribution)
    return bins, log_returns, cumulative_distribution


def fit_power_law(bins, counts):
    """
    Fit a power law to a given set of histogram bins and counts.

    Args:
        bins (np.ndarray): The bins of a histogram binning.
        counts (np.ndarray): The counts of a histogram.

    Returns:
        tuple: The fitted parameters for the power law.

    """
    params, pcov = curve_fit(fit_func, np.log10(bins), np.log10(counts))
    return params


def fit_func(x, a, b):
    """
    Linear function used to fit to data.

    Args:
        x (float | np.ndarray): The x values for the curve.
        a (float | np.ndarray): The slope of the curve.
        b (float | np.ndarray): The y interception of the curve.

    Returns:
        float | np.ndarray: The y value at given point x.

    """
    return a * x + b


if __name__ == "__main__":
    main()
