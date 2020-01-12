from sklearn.linear_model import LogisticRegression
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load():
    X_train = convert_3d_to_2d(np.load('X_train.npy'))
    y_train = np.load('y_train.npy')
    X_test = convert_3d_to_2d(np.load('X_test.npy'))
    y_test = np.load('y_test.npy')

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


print(50, get_precision(50)) # 0.44
print(100, get_precision(100)) # 0.59
print(1000, get_precision(1000)) # 0.772
print(50000, get_precision(50000)) # 0.8857615894039735