import numpy as np
import data_io
import ctypes
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D, Lambda, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


if __name__ == "__main__":

    # pwm_values, accel_readings = data_io.get_data_arrays('../data/raw')
    train_pwm_values, train_accel_values, test_pwm_values, test_accel_values = data_io.get_datasets("../data/raw")
    
    train_accel_values = train_accel_values / 255.0
    train_accel_values = np.abs(np.fft.rfft(train_accel_values, axis=1))
    train_accel_values = np.sum(train_accel_values, axis=-1)
    train_accel_values = np.array(train_accel_values, dtype="float32")

    model_input = Input(shape=train_accel_values.shape[1], dtype="float32")

    model = Sequential(
        (
            model_input,
            Dense(256, activation="relu"), 
            Dense(32, activation="relu"), 
            Dense(1, activation="relu"),
        )
    )

    model.summary()

    # import pdb;pdb.set_trace()
    model.compile(
        optimizer="adam", 
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["accuracy"]
    )

    model.fit(train_accel_values, train_pwm_values, epochs=10000)
    

    import pdb;pdb.set_trace()
