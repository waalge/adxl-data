"""
Set up and provide benchmarking.
"""
import os
import zipfile
import ctypes

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "./data/"
RAW_ZIP = os.path.join(DATA_DIR, "raw.zip")
RAW_DIR = os.path.join(DATA_DIR, "raw")


TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")


def unzip_raw(in_zip=RAW_ZIP, out_dir=DATA_DIR):
    with zipfile.ZipFile(in_zip, "r") as zip_ref:
        zip_ref.extractall(out_dir)


def get_filenames(directory=RAW_DIR):
    return [f for f in os.listdir() if "2020" in f]


def make_test_train_split():
    """
    Make test train split. Write to csvs
    """
    filenames = get_filenames()
    x, y = zip(*[f.split("_") for f in filenames])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=345
    )
    pd.DataFrame({"timestamp": x_train, "pwm": y_train}).to_csv(TRAIN_CSV, index=False)
    pd.DataFrame({"timestamp": x_test, "pwm": y_test}).df.to_csv(TEST_CSV, index=False)


def get_dataset(path_to_csv):
    df = pd.read_csv(path_to_csv)
    files = [os.path.join(RAW_DIR, f"{row.timestamp}_{row.pwm}") for row in df.itertuples()]
    accel_values = np.array([np.fromfile(fx, dtype=ctypes.c_int16).reshape((-1, 3)) for fx in files])
    pwm_values = np.array([int(fx.split("_")[-1]) for fx in files], dtype=ctypes.c_int16)
    return pwm_values, accel_values

def get_train_set():
    return get_dataset(TRAIN_CSV)

def get_test_set():
    return get_dataset(TEST_CSV)

def get_datasets(optional=False):
    return [*get_train_set(), *get_test_set()]
