"""
Set up and provide benchmarking.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = "./data/"


def benchmark_csv(yhat):
    """
    Simple benchmarker.
    May the best yhat win!
    No cheating mind.

    Args:
        yhat (np.array) : array of estimates on test set.

    Returns:
        (int) score in (0,1], with 1 being a perfect score.
    """
    path_to_file = os.path.join(DATA_DIR, "test.csv")
    y = pd.read_csv(path_to_file)["pwm"].to_numpy()
    return sum(1 / (1 + (y - yhat) ** 2)) / y.shape[0]


def demo_benchmark():
    """
    Examples of terrible estimators
    """
    y_random = np.random.randint(0, 255, 2000)
    print("Benchmark y_random : ", benchmark_csv(y_random))
    y_const = np.array([127] * 2000)
    print("Benchmark y_const  : ", benchmark_csv(y_const))


if __name__ == "__main__":
    demo_benchmark()
