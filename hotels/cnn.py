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
import matplotlib.pyplot as plt
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
batch_size = 128


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


def show_confusion_matrix(Y_true, Y_pred):
    conf_matrix = np.zeros((3, 3))
    for i in xrange(len(Y_pred)):
        conf_matrix[Y_true[i], Y_pred[i]] += 1

    norm_conf = []
    for i in conf_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')

    width, height = conf_matrix.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_matrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    cb = fig.colorbar(res)
    class_names = ['negative', 'neutral', 'positive']
    plt.xticks(range(3), class_names)
    plt.yticks(range(3), class_names)

    plt.savefig('confusion_matrix.png', format='png')


def test(X):
    return np.array([np.argmax(model.predict(X[None, i])) for i in xrange(len(X))])


def print_test_msg(Y_true, Y_pred):
    n_total = len(Y_true)
    n_correct = np.sum(Y_pred == Y_true)

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
        model.fit(X_train, Y_train, n_epochs=5, batch_size=batch_size, freq_log=1, freq_checkpoint=500, loss_decay=0.95,
                  checkpoint_dir='./checkpoints')

    if args.mode == 'test' or args.mode == 'debug':
        Y_pred = test(X_dev)
        print_test_msg(Y_dev, Y_pred)
        show_confusion_matrix(Y_dev, Y_pred)

