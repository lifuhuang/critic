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


def prepare_data():
    word_table = Dictionary(glove_path[wv_len]).to_array()[..., None]
    
    df = pd.read_pickle(data_path)  
    df['vector'] = df['vector'].apply(
        lambda x: np.append(x, np.zeros(sentence_len-x.size)))
    split = df.shape[0] // 10 * 9

    # df['c_overall'] = -1.
    # df.loc[df['overall'] <= 3, 'c_overall'] = 0.
    # df.loc[df['overall'] == 5, 'c_overall'] = 1.
    #
    # df['c_cleanliness'] = -1.
    # df.loc[df['cleanliness'] <= 3, 'c_cleanliness'] = 0.
    # df.loc[df['cleanliness'] == 5, 'c_cleanliness'] = 1.
    #
    # df['c_location'] = -1.
    # df.loc[df['location'] <= 3, 'c_location'] = 0.
    # df.loc[df['location'] == 5, 'c_location'] = 1.

    df['overall_p'] = df['overall'] == 5
    df['overall_n'] = df['overall'] <= 4

    df['cleanliness_p'] = df['cleanliness'] == 5
    df['cleanliness_n'] = df['cleanliness'] <= 4

    df['location_p'] = df['location'] == 5
    df['location_n'] = df['location'] <= 4

    X = np.array(list(df['vector']))
    Y = df[['overall_p', 'overall_n', 'cleanliness_p', 'cleanliness_n', 'location_p', 'location_n']].values / 3.0
    
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
    Y = np.random.randint(0, 1, (1000, 3))
    
    X_train = X[:split]
    Y_train = Y[:split]
    X_dev = X[split:]
    Y_dev = Y[split:]
    print 'Completed data preparation!'
    
    return word_table, X_train, Y_train, X_dev, Y_dev


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    args = parser.parse_args()
    
    word_table, X_train, Y_train, X_dev, Y_dev = prepare_data()

    model = NeuralNetwork(
        optimizer=optimizers.Adam(), 
        log_dir='logs', 
        loss_fn='softmax_cross_entropy',
        output_fn='softmax',
        target_dtype='float32')
        
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
    model.add(FullyConnected(6, name='output'))
    model.build()
    print 'Model constructed!'

    try:
        model.load_checkpoint('./checkpoints/')
        print 'checkpoint loaded!'
    except Exception as e:
        print 'File not found!'
    
    if args.mode == 'train':
        model.fit(X_train, Y_train, n_epochs=5, batch_size=32, 
                  freq_log=1, freq_checkpoint=200, loss_decay=0.9, 
                  checkpoint_dir='./checkpoints')
    elif args.mode == 'test':
        n_test = 50
        yh = model.predict(X_dev[:n_test])
        for i in xrange(n_test):
            print '%d:' % i
            print 'Y_true', Y_dev[i]
            print 'Y_pred', yh[i]