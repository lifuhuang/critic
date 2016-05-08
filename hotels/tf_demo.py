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