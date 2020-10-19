# Guess my PWM

This repo includes a dataset of accelerometer data in the presence of a haptic feedback motor.

Aim: Use as a classifier/ estimator benchmarker.

## Setup

The haptic feedback motor and 3D-accelerometer share a solderless breadboard.
Both are connected to a Pi-zero, from which they are controlled and monitored respectively.

The haptic feedback motor is controlled using PWM which varies the effective power that the device receives.
The PWM input rages from 0 (completely on) to 255 (completely off).

The accelerometer is a [adxl345](https://www.analog.com/en/products/adxl345.html) with the following settings:

* 800Hz read rate
* 2g sensitivity settings

Each sample consists of the following:

* A input value between 0 and 255 is generated
* The motor pin is assigned to output PWM with this value
* 800 readings are taken of the accelerometer, each of 6 bytes, 2 for each axis.
WARNING: The polling of the accelerometer by the pi is only approximately equal to the
accelerometer read rate but this was very poorly implemented and maybe a source of unexpected behaviour.
* The 6 bytes are converted to 3 ints which are repeatedly placed in a buffer.
* The buffer is dumped to file, the filename is of the form
```
<"%Y%m%d%H%M%S" formatted timestamp>_<input_value>
```
The motor output is set to off, and a short time period ellapses to ensure the motor is off before the next sample.


## Utils

The `utils/benchmark.py` file contains the functionality to generate the test/ train split, and provide the benchmark score on the test set.

#TODOs

1. Provide baseline model.
