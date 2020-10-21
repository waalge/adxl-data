"""
Run with:

    python main_hist.py

"""

# Environment packages
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense 
import plotille

# project packages
from preprocessing import PreproccessorHist
import data_io

if __name__ == "__main__":

    # Load the training and validation datasets
    # pwm_values, accel_readings = data_io.get_data_arrays('../data/raw')
    train_pwm_values, train_accel_values, test_pwm_values, test_accel_values = data_io.get_datasets("../data/raw")
    
    # Create histograms of the acceleration magnitudes using nbins histrogram bins
    # The bin withss are chosen such that for the training dataset there are an equal number of 
    # entries in each bin
    nbins = 256
    preprocessor = PreproccessorHist(train_accel_values, nbins)
    
    # preprocess the training values and validation values
    pp_train_accel_values = preprocessor.process_block(train_accel_values)
    pp_test_accel_values = preprocessor.process_block(test_accel_values)
    
    # create a model
    model_input = Input(shape=nbins, dtype="int32")
    model = Sequential(
        (
            model_input,
            Dense(64, activation="relu"), 
            Dense(32, activation="relu"), 
            Dense(1, activation="relu"),
        )
    )
    model.summary()

    model.compile(
        optimizer="adam", 
        loss=tf.keras.losses.MeanSquaredError(),
    )
    model.fit(pp_train_accel_values, train_pwm_values, epochs=30, validation_data=(pp_test_accel_values, test_pwm_values))
    
    # create a histogram of abs errors from the validation dataset
    predict_pwm = model.predict(pp_test_accel_values)[:,0]
    predict_diff = np.abs(predict_pwm - test_pwm_values)
    
    print("\nHistogram of ABS errors for validation data:")
    print(plotille.hist(predict_diff))

