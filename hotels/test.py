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

import numpy as np
from legonet import optimizers
from legonet.layers import FullyConnected, Embedding, Convolution, Pooling
from legonet.topology import Sequential, Parallel
from legonet.models import NeuralNetwork

#from utils import Dictionary

data_path = '/mnt/shared/hotels/df1'
glove200d_path = '/mnt/shared/glove/glove.6B.200d.txt'
sentence_len = 64

dataset_size = 1000
dev_ratio = 0.1
dict_size = 4000
wordvec_len = 100
    
def prepare_fake_data():
    
    word_table = np.random.randn(dict_size, wordvec_len, 1)
    X = np.random.randint(0, dict_size, (dataset_size, sentence_len))
    Y = np.random.randint(0, 3, dataset_size)
    split = dataset_size * (1 - dev_ratio)
    X_train = X[:split]
    X_dev = X[split:]
    Y_train = Y[:split]
    Y_dev = Y[split:]
    return word_table, X_train, Y_train, X_dev, Y_dev 
    
if __name__ == '__main__':
    word_table, X_train, Y_train, X_dev, Y_dev = prepare_fake_data()
    
    model = NeuralNetwork(optimizer=optimizers.Adam(), log_dir='logs')
    model.add(Embedding([sentence_len], word_table))

    seq1 = Sequential('3gram')
    seq1.add(Convolution([3, wordvec_len], 64, padding='VALID'))
    seq1.add(Pooling([sentence_len-2, 1], strides=[1, 1]))
    
    seq2 = Sequential('4gram')
    seq2.add(Convolution([4, wordvec_len], 64, padding='VALID'))
    seq2.add(Pooling([sentence_len-3, 1], strides=[1, 1]))
        
    seq3 = Sequential('5gram')
    seq3.add(Convolution([5, wordvec_len], 64, padding='VALID'))
    seq3.add(Pooling([sentence_len-4, 1], strides=[1, 1]))
    
    para = Parallel(along_dim=1)
    para.add(seq1)
    para.add(seq2)
    para.add(seq3)
    
    model.add(para)
    model.add(FullyConnected(128, 'relu'))
    model.add(FullyConnected(3, name='output'))
    #pdb.set_trace()
    model.build()
    print 'Model constructed!'
    
    try:
        model.load_checkpoint('./checkpoints/')
        print 'checkpoint loaded!'
    except Exception as e:
        print 'File not found!'
    
    model.fit(X_train, Y_train, n_epochs=2, batch_size=32, freq_log=10,
              loss_decay=0.9, checkpoint_dir='./checkpoints')
