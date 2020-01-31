import numpy as np
import collections


def load(conv=False):
    X_train = convert_3d_to_2d(np.load('../processed/notMNIST/X_train.npy'), conv) / 255  # with normalization
    y_train = np.load('../processed/notMNIST/y_train.npy')
    X_test = convert_3d_to_2d(np.load('../processed/notMNIST/X_test.npy'), conv) / 255  # with normalization
    y_test = np.load('../processed/notMNIST/y_test.npy')
    # {'G': 17495, 'E': 17446, 'B': 17445, 'A': 17441, 'F': 17382, 'C': 17315, 'D': 17315, 'J': 17256, 'H': 17104, 'I': 15196}
    print(collections.Counter(y_train))

    return X_train, y_train, X_test, y_test


def convert_3d_to_2d(arr, conv):
    return np.reshape(arr, [arr.shape[0], arr.shape[1] * arr.shape[2]]) if conv else np.reshape(arr, [arr.shape[0], arr.shape[1], arr.shape[2], 1])
