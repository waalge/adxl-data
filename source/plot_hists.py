import numpy as np
import data_io
from preprocessing import PreproccessorHist
import sys
import matplotlib.pyplot as plt
from math import ceil, sqrt
from itertools import product


def midpoints(x):
    return (x[1:] + x[:-1]) * 0.5


def widths(x):
    return x[1:] - x[:-1]


if __name__ == "__main__":

    nbins = int(sys.argv[1])
    num_pwm_values = int(sys.argv[2])

    pwm_values, accel_readings = data_io.get_data_arrays("../data/raw")
    pwm_plot_values = np.array(
        np.linspace(int(np.min(pwm_values)), int(np.max(pwm_values)), num_pwm_values), dtype="int"
    )
    preprocessor = PreproccessorHist(accel_readings, nbins)
    pp_accel_readings = preprocessor.process_block(accel_readings)

    bars = midpoints(preprocessor.bins)
    widths = widths(preprocessor.bins)
    
    side_length = ceil(sqrt(num_pwm_values))
    fig, ax = plt.subplots(side_length, side_length, figsize=(4 * side_length, 4 * side_length))
    
    ax_indices = tuple(product(range(side_length), range(side_length)))

    for plotx, pwm_value in enumerate(pwm_plot_values):

        indices = np.where(pwm_values == pwm_value)[0]
        for ix in indices:
            axx = ax_indices[plotx][0]
            axy = ax_indices[plotx][1]

            ax[axx][axy].bar(bars, pp_accel_readings[ix, :], width=widths, color="b", alpha=0.1)
            ax[axx][axy].set_title("PWM = " + str(pwm_value))

    fig.savefig("pp_plot.pdf", bbox_inches="tight")
