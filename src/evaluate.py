import h5py
import numpy as np
import sys
from keras.models import model_from_json

# Read input data
input_text_file = sys.argv[1]

# Read vocabulary
data = np.load('../data/train1.pkl')
vocabulary = data['vocabulary']
punctuation = data['punctuations']
inv_punctuation = dict((value, key) for key, value in punctuation.items())

input_data_size = len(vocabulary)
output_data_size = len(punctuation)

# Load pre trained model
model = model_from_json(open('model_architecture.json').read())
model.load_weights('model_weights.h5')

# Predictions
punctuated_text= ""
with open(input_text_file, 'r') as text_file:
  for line in text_file:
    tokens = line.strip().split()
    if len(tokens) > 0:
      wordvec = np.zeros(shape=(len(tokens)+1,1,len(vocabulary)), dtype=np.int8)
      for i in range(len(tokens)):
        if tokens[i] in vocabulary:
          idx = vocabulary[tokens[i]]
        else:
          idx = vocabulary["<unk>"] #special symbol when out of vocabulary
        wordvec[i,0,idx] = 1
      # end of line
      wordvec[-1,0,vocabulary["<END>"]] = 1

      y = model.predict(wordvec)
      punkts = np.argmax(y, axis=1)
      for p in range(len(punkts)-1):
        if punkts[p] == 0:
          punctuated_text += inv_punctuation[punkts[p]] + tokens[p]
        else:
          punctuated_text += ' ' + inv_punctuation[punkts[p]] + ' ' + tokens[p]
      punctuated_text += ' ' + inv_punctuation[punkts[-1]] + '\n'

print(punctuated_text)
