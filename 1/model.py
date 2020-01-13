from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'loaders')))

from load_not_mnist import load


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