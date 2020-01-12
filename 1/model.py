from sklearn.linear_model import LogisticRegression
import numpy as np
import collections

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load():
    X_train = convert_3d_to_2d(np.load('../processed/notMNIST/X_train.npy')) / 255 # with normalization
    y_train = np.load('../processed/notMNIST/y_train.npy')
    X_test = convert_3d_to_2d(np.load('../processed/notMNIST/X_test.npy')) / 255 # with normalization
    y_test = np.load('../processed/notMNIST/y_test.npy')
    # {'G': 17495, 'E': 17446, 'B': 17445, 'A': 17441, 'F': 17382, 'C': 17315, 'D': 17315, 'J': 17256, 'H': 17104, 'I': 15196}
    print(collections.Counter(y_train))

    return X_train, y_train, X_test, y_test


def convert_3d_to_2d(arr):
    return np.reshape(arr, [arr.shape[0], arr.shape[1] * arr.shape[2]])


X_train, y_train, X_test, y_test = load()


def get_precision(k_samples=50):
    # all parameters not specified are set to their defaults
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train[0:k_samples], y_train[0:k_samples])
    score = logisticRegr.score(X_test[0:k_samples], y_test[0:k_samples])

    return score


"""
If we run without normalization, code will execute much longer and results
will be following: 0.44, 0.59, 0.772, 0.8857615894039735
"""
print(50, get_precision(50)) # 0.52
print(100, get_precision(100)) # 0.59
print(1000, get_precision(1000)) # 0.822
print(50000, get_precision(50000)) # 0.8867229224524674