# coding: utf-8

import numpy as np
import cPickle
import utils
import h5py
import os
import sys


def convert_file(file_path, vocab_file, punct_file, output_path):
    punctuations = {" ":0, ".":1, ",":2}
    punctuations = utils.load_punctuations(punct_file)
    vocabulary = utils.load_vocabulary(vocab_file)
    punctuation = " "
    time_steps = 1 #to be used in future experiments
    
    filename = 'database' # output file name
    f = h5py.File(os.path.join(output_path, filename+'.h5'), "w")
    input_dset = f.create_dataset('inputs', (100, time_steps,len(vocabulary)), dtype='i8', maxshape=(None, time_steps, len(vocabulary)))
    output_dset = f.create_dataset('outputs', (100, len(punctuations)), dtype='i8', maxshape=(None, len(punctuations)))
    data_counter = 0
    with open(file_path, 'r') as corpus:
        for line in corpus:
            array = np.zeros(shape=(1, len(vocabulary)), dtype=np.int8)
            array[0,utils.input_word_index(vocabulary, "<START>")] = 1
            input_dset[data_counter] = array

            array = np.zeros(shape=(1, len(punctuations)), dtype=np.int8)
            array[0,utils.punctuation_index(punctuations, " ")] = 1
            output_dset[data_counter] = array
            data_counter += 1
            if data_counter == input_dset.shape[0]:
                input_dset.resize(input_dset.shape[0]+1000, axis=0)
                output_dset.resize(output_dset.shape[0]+1000, axis=0)

            for token in line.split():
                if token in punctuations:
                    punctuation = token
                    continue
                else:
                    array = np.zeros(shape=(1, len(vocabulary)), dtype=np.int8)
                    array[0,utils.input_word_index(vocabulary, token)] = 1
                    input_dset[data_counter] = array

                    array = np.zeros(shape=(1, len(punctuations)), dtype=np.int8)
                    array[0,utils.punctuation_index(punctuations, punctuation)] = 1
                    output_dset[data_counter] = array

                    punctuation = " "
                    data_counter += 1
                    if data_counter == input_dset.shape[0]:
                        input_dset.resize(input_dset.shape[0]+1000, axis=0)
                        output_dset.resize(output_dset.shape[0]+1000, axis=0)

            array = np.zeros(shape=(1, len(vocabulary)), dtype=np.int8)
            array[0,utils.input_word_index(vocabulary, "<END>")] = 1
            input_dset[data_counter] = array
         
            array = np.zeros(shape=(1, len(punctuations)), dtype=np.int8)
            array[0,utils.punctuation_index(punctuations, punctuation)] = 1
            output_dset[data_counter] = array

            data_counter += 1
            if data_counter == input_dset.shape[0]:
                input_dset.resize(input_dset.shape[0]+1000, axis=0)
                output_dset.resize(output_dset.shape[0]+1000, axis=0)


    input_dset.resize(data_counter, axis=0)
    output_dset.resize(data_counter, axis=0)

    data = {"vocabulary": vocabulary, "punctuations": punctuations, 
           "total_size": data_counter}
    
    with open(os.path.join(output_path, filename+'.pkl'), 'wb') as output_file:
        cPickle.dump(data, output_file, protocol=cPickle.HIGHEST_PROTOCOL)

    print("Done!")


if __name__ == "__main__":

  import argparse
  
  OptParser = argparse.ArgumentParser(description = __doc__)

  OptParser.add_argument("-i", "--input-file", dest='ipath', help="Corpus that will be used for training.")
  OptParser.add_argument("-v", "--vocabulary-file", dest='voc', help="Vocabulary file. One word per line.")
  OptParser.add_argument("-p", "--punctuations-file", dest='punct', help="Punctuations file. One punctuation per line.")
  OptParser.add_argument("-o", "--output-path", dest='opath', help="Path to the foler where the output files will be written.")

  options = OptParser.parse_args()

  if (options.ipath is None) or (options.voc is None) or (options.punct is None) or (options.opath is None):
    OptParser.print_help()
    sys.exit(1)

  convert_file(options.ipath, options.voc, options.punct, options.opath)
