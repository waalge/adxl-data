"""
Set up and provide benchmarking.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "./data/"
RAW_DIR = os.path.join(DATA_DIR, "raw")


def get_filenames():
    return [f for f in os.listdir(RAW_DIR) if "2020" in f]


def make_test_train_split():
    """
    Make test train split. Write to csvs
    """
    filenames = get_filenames()
    x, y = zip(*[f.split("_") for f in filenames])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=345
    )
    df = pd.DataFrame({"timestamp": x_train, "pwm": y_train})
    path_to_file = os.path.join(DATA_DIR, "train.csv")
    df.to_csv(path_to_file, index=False)
    df = pd.DataFrame({"timestamp": x_test, "pwm": y_test})
    path_to_file = os.path.join(DATA_DIR, "test.csv")
    df.to_csv(path_to_file, index=False)


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
    # make_test_train_split
    make_test_train_split()
    demo_benchmark()
