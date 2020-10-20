import numpy as np
import data_io
import ctypes
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D, Lambda, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def histedges_equalN(x, nbin):
    # https://stackoverflow.com/questions/39418380/histogram-with-equal-number-of-points-in-each-bin
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
        np.arange(npt),
        np.sort(x)
    )


class Preproccessor:
    def __init__(self, data, nbins=512):
        self.nbins = nbins
        data = self._reduction(data)
        _, self.bins = np.histogram(data.ravel(), bins=histedges_equalN(data.ravel(), nbins))

    def _reduction(self, data):
        data = np.square(data)
        data = np.sum(data, axis=-1)
        return data
    
    def __call__(self, data):
        data = self._reduction(data)
        hist, _ = np.histogram(data.ravel(), bins=self.bins)
        return hist

    def process_block(self, data):
        ndata = data.shape[0]
        out = np.zeros((data.shape[0], self.nbins), dtype='int32')
        for rowx in range(ndata):
            out[rowx, :] = self(data[rowx, : , :])
        return out


if __name__ == "__main__":

    # pwm_values, accel_readings = data_io.get_data_arrays('../data/raw')
    train_pwm_values, train_accel_values, test_pwm_values, test_accel_values = data_io.get_datasets("../data/raw")
    
    nbins = 256
    preprocessor = Preproccessor(train_accel_values, nbins)
    
    pp_train_accel_values = preprocessor.process_block(train_accel_values)

    model_input = Input(shape=nbins, dtype="int32")

    model = Sequential(
        (
            model_input,
            Dense(256, activation="relu"), 
            Dense(128, activation="relu"), 
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

    model.fit(pp_train_accel_values, train_pwm_values, epochs=10000)
    

    import pdb;pdb.set_trace()
