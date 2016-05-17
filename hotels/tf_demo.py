# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:51:35 2016

@author: lifu
"""
import numpy as np
import tensorflow as tf

batch_size=50
X_train = np.random.randn(50, 200)
y_train = np.random.randint(0, 3, 50)


W = tf.Variable(tf.random_normal([200, 3]), name='weight')
b = tf.Variable(tf.zeros([1, 3]), name='bias')
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, 200], name='X')
Y = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='Y')
Z = tf.matmul(X, W) + b
output = tf.nn.softmax(Z)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(Z, Y))
opt = tf.train.AdamOptimizer()
update = opt.minimize(loss)
loss_summary = tf.scalar_summary('loss_val', loss)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    
    
    train_writer = tf.train.SummaryWriter('/home/lifu/train', sess.graph)
    sess.run(init_op)
    for i in xrange(1000):
        loss_val, _, summary = sess.run(
            [loss, update, loss_summary], 
            feed_dict={X: X_train, Y: y_train})
        if i % 100 == 0:
            train_writer.add_summary(summary, i)
        
        print 'iteration %d: %g' % (i, loss_val)


import sys
######
sys.path.append('/home/lifu/PySpace/dimsum')
######

import os.path as op

import numpy as np
import pandas as pd
from dimsum import optimizers
from dimsum.layers import Dense, Input
from dimsum.models import NeuralNetwork
from dimsum.callbacks import LossPrinter, IterationPrinter, CheckpointSaver

data_path = '/mnt/shared/hotels/df1'
params_path = '/home/lifu/PySpace/critic/hotels/params.npz'

        
if __name__ == '__main__':
    model = NeuralNetwork(objective='cee')
    model.add(Input(200))
    model.add(Dense(200, 128, activation='relu')) 
    model.add(Dense(128, 64, activation='relu'))
    model.add(Dense(64, 3, activation='softmax'))
     
    print 'Model constructed!'
    
    
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
    

    callbacks = [(IterationPrinter(sys.stdout), 100),
                 (LossPrinter(outfd=sys.stdout, update_rate=1), 100), 
                 (LossPrinter(x=X_dev, y=Y_dev, outfd=sys.stdout), 5000),
                 (CheckpointSaver(path='/mnt/shared/cp', verbose=True), 1000)]
    try:
        model.fit(X_train, Y_train, n_epochs=10, batch_size=64, use_checkpoint='/mnt/shared/cp',
                  optimizer=optimizers.Adagrad(0.1), callbacks=callbacks)
    except KeyboardInterrupt:
        print 'Manually terminated!'
    
    n_test = 30
    text = df_dev.iloc[:n_test]['text']
    yh = model.predict(X_dev[:n_test])
    for i in xrange(n_test):
        print '%d:' % i
        print 'Y_true', Y_dev[i]
        print 'Y_pred', yh[i]
        print 'text', text.iat[i]