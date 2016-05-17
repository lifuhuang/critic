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
from legonet.layers import FullyConnected, Input, Embedding, Sequential, Parallel
from legonet.models import NeuralNetwork

data_path = '/mnt/shared/hotels/df1'
params_path = '/home/lifu/PySpace/critic/hotels/params.npz'

        
if __name__ == '__main__':
    model = NeuralNetwork(optimizer=optimizers.Adam(), log_dir='logs')
    model.add(Input('input', 200))   
    model.add(FullyConnected('hidden1', 512, 'relu'))    
    model.add(FullyConnected('hidden2', 256, 'relu'))
    model.add(FullyConnected('hidden3', 128, 'relu'))
    model.add(FullyConnected('output', 2))
    model.build()
    print 'Model constructed!'
    
    try:
        model.load_checkpoint('./checkpoints/')
        print 'checkpoint loaded!'
    except Exception as e:
        print 'File not found!'
    
    df = pd.read_pickle(data_path)    
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
    
    model.fit(X_train, Y_train, n_epochs=2, batch_size=128, 
              loss_decay=0.9, checkpoint_dir='./checkpoints')
    
    n_test = 30
    text = df_dev.iloc[:n_test]['text']
    yh = model.predict(X_dev[:n_test])
    for i in xrange(n_test):
        print '%d:' % i
        print 'Y_true', Y_dev[i]
        print 'Y_pred', yh[i]
        print text.iat[i]