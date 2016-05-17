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
######
sys.path.append('/home/lifu/PySpace/legonet')
######

import os.path as op

import tensorflow as tf
import numpy as np
import pandas as pd
from legonet import optimizers
from legonet.layers import FullyConnected, Embedding, Convolution2D, Pooling2D
from legonet.topology import Sequential, Parallel
from legonet.models import NeuralNetwork

from utils import Dictionary

data_path = '/mnt/shared/hotels/df1'
glove200d_path = '/mnt/shared/glove/glove.6B.200d.txt'
sentence_len = 256


def prepare_data():
    word_table = Dictionary(glove200d_path).to_array()[..., None]
    
    df = pd.read_pickle(data_path)  
    df = df[df['vector'].notnull() & df['ratings.overall'].notnull()]
    df = df[df['vector'].apply(len) <= 256]
    df['vector'] = df['vector'].apply(
        lambda x: np.append(x, np.zeros(sentence_len-x.size)))
    split = df.shape[0] // 10 * 9
        
    target = df['ratings.overall']
    df['negative'] = (target <= 3).astype(int)
    df['neutral'] = ((target > 3) & (target < 5)).astype(int)
    df['positive'] = (target == 5).astype(int)
    
    df_train = df.iloc[:split, :]
    X_train = np.vstack(df_train.loc[:, 'vector'])
    Y_train = df_train.loc[:, ['negative', 'neutral', 'positive']].values
    
    
    df_dev = df.iloc[split:, :]
    X_dev = np.vstack(df_dev.loc[:, 'vector'])
    Y_dev = df_dev.loc[:, ['negative', 'neutral', 'positive']].values
    
    return word_table, X_train, Y_train, X_dev, Y_dev
    
if __name__ == '__main__':
    
    word_table, X_train, Y_train, X_dev, Y_dev = prepare_data()
    
    model = NeuralNetwork(optimizer=optimizers.Adam(), log_dir='logs')
    model.add(Embedding('embedding', [sentence_len], word_table))  # b, l, 200, 1

    seq1 = Sequential('seq1')
    seq1.add(Convolution2D('conv_3gram', [3, 200], 50)) # b, l, 1, 50
    seq1.add(Pooling2D('pooling_3gram', [sentence_len, 200], strides=[1, 1])) # b, 1, 1, 50
    
    seq2 = Sequential('seq2')
    seq2.add(Convolution2D('conv_3gram', [3, 200], 50)) # b, l, 1, 50
    seq2.add(Pooling2D('pooling_3gram', [sentence_len, 200], strides=[1, 1])) # b, 1, 1, 50
    
    para = Parallel('parallel', along_dim=3)
    para.add(seq1)
    para.add(seq2)
    
    model.add(para)
    model.add(FullyConnected('fc', 128, 'relu'))    
    model.add(FullyConnected('output', 3))
    model.build()
    print 'Model constructed!'
    
    try:
        model.load_checkpoint('./checkpoints/')
        print 'checkpoint loaded!'
    except Exception as e:
        print 'File not found!'
    

#   X_train, Y_train, X_dev, Y_dev = prepare_data()
    
    model.fit(X_train, Y_train, n_epochs=2, batch_size=128, freq_log=10,
              loss_decay=0.9, checkpoint_dir='./checkpoints')
    
    n_test = 30
    text = df_dev.iloc[:n_test]['text']
    yh = model.predict(X_dev[:n_test])
    for i in xrange(n_test):
        print '%d:' % i
        print 'Y_true', Y_dev[i]
        print 'Y_pred', yh[i]
        print text.iat[i]