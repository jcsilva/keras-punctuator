# coding: utf-8

import numpy as np
import cPickle
import utils
import h5py
import os

def convert_files(file_paths, vocabulary, punctuations, use_pauses, output_path):
    inputs = []
    outputs = []
    punctuation = " "
    pause = 0.
    
    if use_pauses:
        pauses = []

    for file_path in file_paths:
        with open(file_path, 'r') as corpus:
            for line in corpus:
                for token in line.split():
                    if token in punctuations:
                        punctuation = token
                        continue
                    elif token.startswith("<sil="):
                        pause = float(token.replace("<sil=","").replace(">",""))
                        continue
                    else:
                        array = np.zeros(shape=(1, len(vocabulary)), dtype=np.int8)
                        array[0,utils.input_word_index(vocabulary, token)] = 1
                        inputs.append(array)
                        array = np.zeros(shape=(1, len(punctuations)), dtype=np.int8)
                        array[0,utils.punctuation_index(punctuations, punctuation)] = 1
                        outputs.append(array)
                        if use_pauses:
                            pauses.append(pause)
                        punctuation = " "
                        pause = 0.

                array = np.zeros(shape=(1, len(vocabulary)), dtype=np.int8)
                array[0,utils.input_word_index(vocabulary, "<END>")] = 1
                inputs.append(array)

                array = np.zeros(shape=(1, len(punctuations)), dtype=np.int8)
                array[0,utils.punctuation_index(punctuations, punctuation)] = 1
                outputs.append(array)

#    if use_pauses:
#        pauses.append(pause)

    assert len(inputs) == len(outputs)

    inputs = np.array(inputs, dtype=np.int8).reshape((len(inputs), 1, len(vocabulary)))
    outputs = np.array(outputs, dtype=np.int16).reshape((len(inputs), len(punctuations)))
##    if use_pauses:
##       pauses = np.array(pauses, dtype=np.float32) #[:batch_size*num_batches].reshape((batch_size, num_batches)).T

    total_size = len(inputs) #batch_size*num_batches

    f = h5py.File(output_path + '.h5', "w")
    dset = f.create_dataset('inputs', data=inputs, dtype='i8')
    dset = f.create_dataset('outputs',data=outputs, dtype='i8')

    data = {"vocabulary": vocabulary, "punctuations": punctuations, 
           "total_size": total_size}
    
##    if use_pauses:
##        data["pauses"] = pauses

    with open(output_path + '.pkl', 'wb') as output_file:
        cPickle.dump(data, output_file, protocol=cPickle.HIGHEST_PROTOCOL)


PHASE1_TRAIN_PATH = "../data/train1"
PHASE1_DEV_PATH = "../data/dev1"
PUNCTUATIONS = {" ": 0, ".PERIOD": 1, ",COMMA": 2}
VOCABULARY_FILE = "../raw_data/vocab"
TRAIN_DATA = "../raw_data/train.txt"
DEV_DATA = "../raw_data/dev.txt"


if not os.path.exists("../data"):
  os.makedirs("../data")

print "Converting data...\n"

vocabulary = utils.load_vocabulary(VOCABULARY_FILE)

convert_files([TRAIN_DATA], vocabulary, PUNCTUATIONS,  False, PHASE1_TRAIN_PATH)

convert_files([DEV_DATA], vocabulary, PUNCTUATIONS, False, PHASE1_DEV_PATH)

