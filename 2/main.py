from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout
from keras import optimizers
import pandas as pd
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'loaders')))

from load_not_mnist import load


X_train, y_train, X_test, y_test = load()

with tf.device('/cpu:0'):
    # definition
    model = Sequential()

    # adding layers
    model.add(Dense(200, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='hidden_0'))
    model.add(Dropout(0.1, name='dropout_0'))
    model.add(Dense(170, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='hidden_1'))
    model.add(Dropout(0.1, name='dropout_1'))
    model.add(Dense(130, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='hidden_2'))
    model.add(Dropout(0.1, name='dropout_2'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='hidden_3'))
    model.add(Dropout(0.1, name='dropout_3'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu', name='hidden_4'))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])

    # train
    model.fit(X_train, np.array(pd.get_dummies(y_train)),
              epochs=1000,
              verbose=1,
              validation_data=(X_test, np.array(pd.get_dummies(y_test))))

    score, acc = model.evaluate(X_test, np.array(pd.get_dummies(y_test)), verbose=1)
    print(score, acc) # 0.92, 0.0328 - 60 epochs