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


data = np.load('../data/train1.pkl')
total_size = data['total_size']
vocabulary = data['vocabulary']
punctuation = data['punctuations']

input_data_size = len(vocabulary)
output_data_size = len(punctuation)

# Adapted from https://github.com/fchollet/keras/issues/68#issuecomment-94881051
def load_data(datapath, train_start, n_training_examples, test_start, n_test_examples):
    X_train = HDF5Matrix(datapath, 'inputs', train_start, train_start+n_training_examples)
    y_train = HDF5Matrix(datapath, 'outputs', train_start, train_start+n_training_examples)
    X_test = HDF5Matrix(datapath, 'inputs', test_start, test_start+n_test_examples)
    y_test = HDF5Matrix(datapath, 'outputs', test_start, test_start+n_test_examples)
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data('../data/train1.h5', 
                                   0, int(math.floor(0.85*total_size)), 
                                   int(math.ceil(0.85*total_size)), 
                                   int(total_size-math.ceil(0.85*total_size)))


print('Build model...')

hidden_neurons = 100

model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(1, input_data_size), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(hidden_neurons, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_data_size, init="uniform"))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


earlyStopping=EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

hist = model.fit(X_train, y_train, 
                 batch_size=200, 
                 nb_epoch=15, 
                 validation_data=(X_test, y_test), 
                 show_accuracy=True, 
                 callbacks=[earlyStopping], 
                 shuffle='batch')

score, acc = model.evaluate(X_test, y_test,
                            batch_size=16,
                            show_accuracy=True)

print('Test score:', score)
print('Test accuracy:', acc)
print(hist.history)





#def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
#    a = np.log(a) / temperature
#    a = np.exp(a) / np.sum(np.exp(a))
#    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
#for iteration in range(1,7001):
#    print()
#    print('-' * 50)
#    print('Iteration', iteration)

#    model.fit(X, y)

#    start_index = random.randint(0, len(text) - num_unrollings -1)


#    if (iteration % 1000) == 0:
#        for diversity in [0.2, 0.5, 1.0, 1.2]:
#            print()
#            print('----- diversity:', diversity)

#            generated = ''
#            sentence = text[start_index]
#            generated += sentence
#            print('----- Generating with seed: "' + sentence + '"')
#            sys.stdout.write(generated)
#
#            for i in range(400):
#                x = np.zeros((1, vocabulary_size))
#                for t, char in enumerate(sentence):
#                    x[0, char2id(char)] = 1.
#
#                preds = model.predict(x, verbose=0)[0]
#                next_index = sample(preds, diversity)
#                next_char = id2char(next_index)
#
#                generated += next_char
#                sentence = sentence[1:] + next_char

#                sys.stdout.write(next_char)
#                sys.stdout.flush()
#            print()
