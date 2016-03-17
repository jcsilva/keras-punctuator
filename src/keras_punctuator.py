from __future__ import print_function
import math
import numpy as np
import random
import string
import sys
import h5py

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.utils.io_utils import HDF5Matrix
from sklearn.metrics import classification_report

data = np.load('../data/train1.pkl')
total_size  = data['total_size']
vocabulary  = data['vocabulary']
punctuation = data['punctuations']

input_data_size  = len(vocabulary)
output_data_size = len(punctuation)
train_amount     = 0.85

# Adapted from https://github.com/fchollet/keras/issues/68#issuecomment-94881051
def load_data(datapath, train_start, n_training_examples, test_start, n_test_examples):
    X_train = HDF5Matrix(datapath, 'inputs',  train_start, train_start+n_training_examples)
    y_train = HDF5Matrix(datapath, 'outputs', train_start, train_start+n_training_examples)
    X_test  = HDF5Matrix(datapath, 'inputs',  test_start,  test_start+n_test_examples)
    y_test  = HDF5Matrix(datapath, 'outputs', test_start,  test_start+n_test_examples)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data('../data/train1.h5', 
                                   0, int(math.floor(train_amount*total_size)), 
                                   int(math.ceil(train_amount*total_size)), 
                                   int(total_size-math.ceil(train_amount*total_size)))

print('Build model...')

hidden_neurons = 100

model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(1, input_data_size), return_sequences=False))
model.add(Dropout(0.2))
#model.add(LSTM(hidden_neurons, return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(output_data_size, init="uniform"))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

earlyStopping=EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

# Train 
hist = model.fit(X_train, y_train, 
                 batch_size=200, 
                 nb_epoch=150, 
                 validation_data=(X_test, y_test), 
                 show_accuracy=True, 
                 callbacks=[earlyStopping], 
                 shuffle='batch')

# Predictions
y_pred = model.predict(X_test)

# Classification report
y_gen = np.argmax(y_pred, axis=1)
y_ref = list()
for i in y_test:
  y_ref.append(np.argmax(i))

print(classification_report(y_ref, y_gen))

# Save model and weights
json_string = model.to_json()
with open('model_architecture.json', 'w') as output_file:
  output_file.write(json_string)
model.save_weights('model_weights.h5', overwrite=True)
