import os

from models.v1 import main_hist

from utils import data_io

def setup():
    if not os.path.isfile(data_io.RAW_ZIP):
        raise FileNotFoundError("Can't find raw.zip file of dataset")
    if not os.path.isdir(data_io.RAW_DIR):
        print("Unzipping raw...")
        data_io.unzip_raw()
    if not os.path.isfile(data_io.TRAIN_CSV):
        print("Making test train split...")
        data_io.make_test_train_split()


if __name__ == "__main__":
    setup()
    main_hist.run()
