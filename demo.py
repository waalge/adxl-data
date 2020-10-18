#!/usr/bin/python3
"""
Example of reading data, FT and plotting
"""

import os
import zipfile
import ctypes

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


DATA_DIR = "./data/"
RAW_ZIP = os.path.join(DATA_DIR, "raw.zip")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def get_array(path_to_file):
    """
    Args:
        path_to_file (str) : path to binary file

    Returns:
        (:obj:`np.array`) of shape [800, 3] where the last index corresponds to signal in x,y, and z dimensions respectively
        """
    with open(path_to_file, "rb") as fh:
        vals = fh.read()
    return np.frombuffer(vals, dtype=ctypes.c_int16).reshape((-1, 3))


def plot_window(signal, N):
    """Note that we drop 0th ft term"""
    T = 1.0 / N
    tf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2 - 1)
    plt.plot(tf, 2.0 / N * np.abs(signal[1 : N // 2]))


def show_graph(path_to_file):
    """Make a phase space graph from time slices of readings"""
    arr = get_array(path_to_file)
    N = 400
    shift = N // 4
    for ii in range((4 * arr.shape[0] - 3 * N) // (4 * shift)):
        offset = ii * shift
        signal = sp.fft(arr[offset : offset + N, 2].reshape(-1))
        plot_window(signal, N)
    plt.grid()
    plt.title(path_to_file.split("/")[-1])
    plt.legend()
    plt.show()
    return arr[-10:, :]


def get_filenames(sort=False):
    """If sort, sort by signals with most power"""
    filenames = [f for f in os.listdir(RAW_DIR) if "2020" in f]
    if not sort:
        np.random.shuffle(filenames)
        return filenames
    return sorted(filenames, key=(lambda x: int(x.split("_")[-1])))


def main():
    """
    Read file and plot fts of signal windows.
    """
    print("Use CTRL+C to kill")
    filenames = get_filenames()
    for filename in filenames:
        path_to_file = os.path.join(RAW_DIR, filename)
        print(filename)
        show_graph(path_to_file)


def unzip_raw():
    with zipfile.ZipFile(RAW_ZIP, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)


if __name__ == "__main__":
    if not os.path.isdir(RAW_DIR):
        unzip_raw()
    main()
