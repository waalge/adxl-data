import numpy as np
import glob
import ctypes
import os


def get_data_arrays(directory):
    """
    Load the pwm values and accelerometer values from a directory as int16_t
    """

    files = glob.glob(os.path.join(directory, "*"))
    accel_values = np.array(
        [np.fromfile(fx, dtype=ctypes.c_int16).reshape((-1, 3)) for fx in files]
    )
    pwm_values = np.array(
        [int(fx.split("_")[-1]) for fx in files], dtype=ctypes.c_int16
    )

    return pwm_values, accel_values


def get_datasets(directory, train_ratio=0.9):
    """
    Returns train_pwm_values, train_accel_values, test_pwm_values and
    test_accel_values from a directory.
    """
    pwm_values, accel_values = get_data_arrays(directory)

    N = pwm_values.shape[0]
    N_train = int(train_ratio * N)
    indices = np.random.permutation(N)

    train_indices = indices[:N_train]
    test_indices = indices[N_train:]

    return (
        pwm_values[train_indices],
        accel_values[train_indices, :, :],
        pwm_values[test_indices],
        accel_values[test_indices, :, :],
    )
