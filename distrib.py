"""
Plot frequency of powers
"""

import os

import matplotlib.pyplot as plt


DATA_DIR = "./data/"
RAW_DIR = os.path.join(DATA_DIR, "raw")


def main():
    dist = [0] * 256
    for filename in sorted(os.listdir(RAW_DIR)):
        if "2020" in filename:
            num = int(filename.split("_")[-1])
            dist[num] += 1
    plt.bar(list(range(256)), dist)
    plt.show()


if __name__ == "__main__":
    print("If fail, run demo first")
    main()
