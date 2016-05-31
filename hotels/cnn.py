# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:49:41 2016

@author: lifu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:49:42 2016

@author: lifu
"""

import sys
import argparse

######
sys.path.append('/home/lifu/PySpace/legonet')
######

import numpy as np
import pandas as pd
from legonet import optimizers
from legonet.layers import FullyConnected, Embedding, Convolution, Pooling
from legonet.pieces import Sequential, Parallel
from legonet.models import NeuralNetwork

from utils import Dictionary

data_path = '/home/lifu/hotels/regression_dataset'
glove_path = {200: '/home/lifu/glove/glove.6B.200d.txt', 
              100: '/home/lifu/glove/glove.6B.100d.txt'}
sentence_len = 256
wv_len = 200
batch_size = 64


def prepare_data():
    word_table = Dictionary(glove_path[wv_len]).to_array()[..., None]
    
    df = pd.read_pickle(data_path)  
    df['vector'] = df['vector'].apply(
        lambda x: np.append(x, np.zeros(sentence_len-x.size)))
    split = df.shape[0] // 10 * 9

    df['c_overall'] = 1
    df.loc[df['overall'] <= 3, 'c_overall'] = 0.
    df.loc[df['overall'] == 5, 'c_overall'] = 2.

    X = np.array(list(df['vector']))
    Y = df['c_overall'].values
    
    X_train = X[:split]
    Y_train = Y[:split]
    X_dev = X[split:]
    Y_dev = Y[split:]

    print 'Completed data preparation!'
    
    return word_table, X_train, Y_train, X_dev, Y_dev


def prepare_data_deprecated():
    dict_size = 10000
    word_table = np.random.randn(dict_size, 200, 1)
    split = 900

    X = np.random.randint(0, dict_size, (1000, sentence_len))
    Y = np.random.randint(0, 3, (1000,))
    
    X_train = X[:split]
    Y_train = Y[:split]
    X_dev = X[split:]
    Y_dev = Y[split:]
    print 'Completed data preparation!'
    
    return word_table, X_train, Y_train, X_dev, Y_dev


def print_test(X, Y):
    n_correct = 0
    n_total = len(X)
    for i in xrange(len(X)):
        yh = np.argmax(model.predict(X[None, i]))
        if yh == Y[i]:
            n_correct += 1

    print 'Total samples: {n_samples}\n' \
          'Correct predictions: {n_correct}\n' \
          'Accuracy: {precesion}%'.format(n_samples=n_total, n_correct=n_correct,
                                          precesion=(n_correct * 100.0 / n_total))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    args = parser.parse_args()

    if args.mode == "debug":
        word_table, X_train, Y_train, X_dev, Y_dev = prepare_data_deprecated()
    else:
        word_table, X_train, Y_train, X_dev, Y_dev = prepare_data()

    model = NeuralNetwork(
        optimizer=optimizers.Adam(), 
        log_dir='logs', 
        loss_fn='sparse_softmax_cross_entropy',
        output_fn='softmax',
        target_dtype='int64')
        
    model.add(Embedding([sentence_len], word_table))

    seq1 = Sequential('3gram')
    seq1.add(Convolution([3, wv_len], 64, padding='VALID'))
    seq1.add(Pooling([sentence_len - 2, 1], strides=[1, 1]))
    
    seq2 = Sequential('4gram')
    seq2.add(Convolution([4, wv_len], 64, padding='VALID'))
    seq2.add(Pooling([sentence_len - 3, 1], strides=[1, 1]))
        
    seq3 = Sequential('5gram')
    seq3.add(Convolution([5, wv_len], 64, padding='VALID'))
    seq3.add(Pooling([sentence_len - 4, 1], strides=[1, 1]))
    
    para = Parallel(along_dim=3)
    para.add(seq1)
    para.add(seq2)
    para.add(seq3)
    
    model.add(para)
    model.add(FullyConnected(128, 'relu'))
    model.add(FullyConnected(3, name='output'))
    model.build()
    print 'Model constructed!'

    try:
        model.load_checkpoint('./checkpoints/')
        print 'checkpoint loaded!'
    except Exception as e:
        print 'File not found!'
    
    if args.mode == 'train' or args.mode == 'debug':
        model.fit(X_train, Y_train, n_epochs=5, batch_size=batch_size, freq_log=1, freq_checkpoint=200, loss_decay=0.9,
                  checkpoint_dir='./checkpoints')

    if args.mode == 'test' or args.mode == 'debug':
        #print_test(X_train, Y_train)
        print_test(X_dev, Y_dev)
