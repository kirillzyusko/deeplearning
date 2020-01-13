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
model.add(Dense(200, kernel_initializer='normal', activation='relu', name='hidden_0'))
model.add(Dense(140, kernel_initializer='normal', activation='relu', name='hidden_1'))
model.add(Dense(100, kernel_initializer='normal', activation='relu', name='hidden_2'))
model.add(Dense(70, kernel_initializer='normal', activation='relu', name='hidden_3'))
model.add(Dense(10, kernel_initializer='normal', activation='relu', name='hidden_4'))

# compile model
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

# train
model.fit(X_train, np.array(pd.get_dummies(y_train)),
          epochs=50,
          verbose=1,
          validation_data=(X_test, np.array(pd.get_dummies(y_test))))

score, acc = model.evaluate(X_test, np.array(pd.get_dummies(y_test)), verbose=1)
print(score, acc) # 0.92, 0.0210 - 37 epochs