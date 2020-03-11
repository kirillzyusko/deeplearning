from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import pandas as pd
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'loaders')))

from load_not_mnist import load


X_train, y_train, X_val, y_val, X_test, y_test = load(conv=True)

with tf.device('/cpu:0'):
    # definition
    model = Sequential()

    # adding layers
    model.add(Conv2D(6, kernel_size=(3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    # train
    model.fit(X_train, np.array(pd.get_dummies(y_train)),
              epochs=5,
              verbose=1,
              validation_data=(X_val, np.array(pd.get_dummies(y_val))))

    score, acc = model.evaluate(X_test, np.array(pd.get_dummies(y_test)), verbose=1)
    print(score, acc)
