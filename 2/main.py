from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'loaders')))

from load_not_mnist import load


X_train, y_train, X_test, y_test = load()

# definition
model = Sequential()

# adding layers
#model.add(Dense(500, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hidden_0'))
#model.add(Dense(300, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hidden_1'))
#model.add(Dense(200, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hidden_2'))
#model.add(Dense(25, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hidden_3'))
model.add(Dense(10, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hidden_4'))

# compile model
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# train
model.fit(X_train[0:50000], np.array(pd.get_dummies(y_train[0:50000])),
          epochs=20,
          verbose=1,
          validation_data=(X_test[0:50000], np.array(pd.get_dummies(y_test[0:50000]))))

score, acc = model.evaluate(X_test[0:50000], np.array(pd.get_dummies(y_test[0:50000])), verbose=1)
print(score, acc)