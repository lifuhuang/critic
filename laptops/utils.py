# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:24:03 2016

@author: lifu
"""

import numpy as np
from nltk.tokenize import word_tokenize

def stream_process(source, dest, func, pred=None, report_freq=None):
    counter = 0
    with open(source, 'r') as fr:
        with open(dest, 'w') as fw:
            for line in fr:
                counter += 1
                if (report_freq is not None and 
                    counter % report_freq == 0):
                    print('%d lines processed!')
                if pred is None or pred(line):
                    fw.write(func(line))

def embed_sentence(text, dictionary):
    vecs = [dictionary[word] for word in word_tokenize(text) 
                                if word in dictionary]
    if vecs:
        return np.mean(np.array(vecs), axis=0)
    else:
        return np.zeros(next(dictionary.itervalues()).shape)

def load_word_vectors(path):
    print 'Start loading word vectors...'
    dictionary = {}
    count = 0
    with open(path, 'r') as fp:
        for line in fp:
            count += 1
            segs = line.split()
            word = segs[0]
            vector = np.array(map(float, segs[1:]))
            dictionary[word] = vector
            if count % 100000 == 0:
                print 'Parsed %d lines.' % count
    print 'Completed! Established a dictionary with size: %d' % len(dictionary)
    return dictionary
                
def make_Y(source, dest):
    items = ['review/appearance', 
             'review/aroma',
             'review/palate',
             'review/taste',
             'review/overall']

    scores = []
    count = 0         
    with open(source) as fr:
        for line in fr:
            count += 1
            if count % 100000 == 0:
                print 'Parsed %d lines.' % count
            for item in items:
                if line.startswith(item):
                    scores.append(float(line[len(item)+1: -1]))
    
    assert(len(scores) % 5 == 0)
    Y = []
    
    print 'Generating ndarray...',
    for i in xrange(0, len(scores), 5):
        Y.append(scores[i: i+5])
    
    Y = np.array(Y)
    print 'Done!'
    print 'Generated a ndarray with shape %s' % (Y.shape, )
    np.savetxt(dest, Y)
    return
                        
def make_X(source, dest, dictionary):
    print 'Start making X...'
    count = 0 
    X = []
    with open(source) as fr:
        for line in fr:
            count += 1
            head = 'review/text'
            if line.startswith(head):
                sentence = line[len(head)+1: -1]
                vector = embed_sentence(sentence, dictionary)
                X.append(vector)
            if count % 1000000 == 0:
                print 'Parsed %d lines.' % count            
    print 'Generating ndarray...',
    X = np.array(X)
    print 'Done!'
    print 'Generated a ndarray with shape %s' % (X.shape, )
    np.savetxt(dest, X)
    return
    
def make_X_from_text(source, dest, dictionary):
    count = 0 
    X = []
    with open(source) as fr:
        for line in fr:
            count += 1
            head = 'review/text'
            sentence = line[len(head)+2: -1]
            X.append(embed_sentence(sentence, dictionary))
            if count % 10000 == 0:
                print 'Parsed %d lines.' % count
            if count % 100000 == 0:
                idx = count / 1000000
                print 'Generating ndarray...',
                X = np.array(X)
                print 'Done!'
                print 'Generated a ndarray with shape %s' % (X.shape, )
                np.savetxt(dest + str(idx), X)
                X = []
    return

def clean_text(text):
    """Preprocess text.
    """
    
    # decapitalize
    text = text.lower()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            