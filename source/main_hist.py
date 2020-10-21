import numpy as np
import data_io
import ctypes
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D, Lambda, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from preprocessing import PreproccessorHist


if __name__ == "__main__":

    # pwm_values, accel_readings = data_io.get_data_arrays('../data/raw')
    train_pwm_values, train_accel_values, test_pwm_values, test_accel_values = data_io.get_datasets("../data/raw")
    
    nbins = 256
    preprocessor = PreproccessorHist(train_accel_values, nbins)
    
    pp_train_accel_values = preprocessor.process_block(train_accel_values)
    pp_test_accel_values = preprocessor.process_block(test_accel_values)

    model_input = Input(shape=nbins, dtype="int32")

    model = Sequential(
        (
            model_input,
            Dense(256, activation="softmax"), 
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

    model.fit(pp_train_accel_values, train_pwm_values, epochs=10000, validation_data=(pp_test_accel_values, test_pwm_values))
    

    import pdb;pdb.set_trace()
