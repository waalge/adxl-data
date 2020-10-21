import numpy as np
import data_io
import ctypes

def histedges_equalN(x, nbin):
    # https://stackoverflow.com/questions/39418380/histogram-with-equal-number-of-points-in-each-bin
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


class PreproccessorHist:
    def __init__(self, data, nbins=512):
        self.nbins = nbins
        data = self._reduction(data)
        self.bins = histedges_equalN(data.ravel(), nbins)

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
        out = np.zeros((data.shape[0], self.nbins), dtype="int32")
        for rowx in range(ndata):
            out[rowx, :] = self(data[rowx, :, :])
        return out
