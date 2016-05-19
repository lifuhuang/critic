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
from legonet.layers import FullyConnected, Embedding, Convolution2D, Pooling2D
from legonet.topology import Sequential, Parallel
from legonet.models import NeuralNetwork

from utils import Dictionary

data_path = '/home/lifu/hotels/df1'
glove200d_path = '/home/lifu/glove/glove.6B.200d.txt'
glove100d_path = '/home/lifu/glove/glove.6B.100d.txt'
sentence_len = 256


def prepare_data():
    word_table = Dictionary(glove200d_path).to_array()[..., None]
    
    df = pd.read_pickle(data_path)  
    df = df[df['vector'].notnull() & df['ratings.overall'].notnull()]
    df = df[df['vector'].apply(len) <= sentence_len]
    df['vector'] = df['vector'].apply(
        lambda x: np.append(x, np.zeros(sentence_len-x.size)))
    split = df.shape[0] // 10 * 9
        
    df['target'] = 1
    df.loc[df['ratings.overall'] <= 3, 'target'] = 0
    df.loc[df['ratings.overall'] == 5, 'target'] = 2
    
    X = np.array(list(df['vector']))
    Y = df['target'].values
    
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

    model = NeuralNetwork(optimizer=optimizers.Adam(), log_dir='logs')
    model.add(Embedding([sentence_len], word_table))

    seq1 = Sequential('3gram')
    seq1.add(Convolution2D([3, 100], 64))
    seq1.add(Pooling2D([sentence_len, 100], strides=[1, 1]))
    
    seq2 = Sequential('4gram')
    seq2.add(Convolution2D([5, 100], 64))
    seq2.add(Pooling2D([sentence_len, 100], strides=[1, 1]))
        
    seq3 = Sequential('5gram')
    seq3.add(Convolution2D([5, 100], 64))
    seq3.add(Pooling2D([sentence_len, 100], strides=[1, 1]))
    
    para = Parallel(along_dim=3)
    para.add(seq1)
    para.add(seq2)
    
    model.add(para)
    model.add(FullyConnected(64, 'relu'))
    model.add(FullyConnected(3, name='output'))
    model.build()
    print 'Model constructed!'

    try:
        model.load_checkpoint('./checkpoints/')
        print 'checkpoint loaded!'
    except Exception as e:
        print 'File not found!'
    
    if args.mode == 'train':
        model.fit(X_train, Y_train, n_epochs=2, batch_size=32, 
                  freq_log=10, freq_checkpoint=100, loss_decay=0.9, 
                  checkpoint_dir='./checkpoints')
    elif args.mode == 'test':
        n_test = 30
        yh = model.predict(X_dev[:n_test])
        for i in xrange(n_test):
            print '%d:' % i
            print 'Y_true', Y_dev[i]
            print 'Y_pred', yh[i]