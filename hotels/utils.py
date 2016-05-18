# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:48:41 2016

@author: lifu
"""

import simplejson as json

from pandas.io.json import json_normalize
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np


data_path = '/mnt/shared/hotels/split1'
dest_path = '/mnt/shared/hotels/df1'
glove100d_path = '/mnt/shared/glove/glove.6B.100d.txt'
glove200d_path = '/mnt/shared/glove/glove.6B.200d.txt'

#def embed_sentence(text, dictionary, preprocess=str.lower):
#    text = preprocess(text)
#    vecs = [dictionary[word] for word in word_tokenize(text) 
#                                if word in dictionary]
#    if vecs:
#        return np.mean(np.array(vecs), axis=0)
#    else:
#        return np.zeros(next(dictionary.itervalues()).shape)
#
#def load_word_vectors(path):
#    print 'Start loading word vectors...'
#    dictionary = {}
#    count = 0
#    with open(path, 'r') as fp:
#        for line in fp:
#            count += 1
#            segs = line.split()
#            word = segs[0]
#            vector = np.array(map(float, segs[1:]))
#            dictionary[word] = vector
#            if count % 100000 == 0:
#                print 'Parsed %d lines.' % count
#    print 'Completed! Established a dictionary with size: %d' % len(dictionary)
#    return dictionary
#    
#def make_data_frame_deprecated(source, dest, embedding):
#    with open(source) as fp:
#        df = json_normalize(map(json.loads, fp))
#    # filter non-ascii reviews
#    df = df[df['text'].apply(lambda x: isinstance(x, str))]
#    df['vector'] = df['text'].apply(embedding)
#    pd.to_pickle(df, dest)

class Dictionary(object):
    """ # TODO: docstring
    """
    
    def __init__(self, path):
        """
        """
        
        self.indices = {'<NUL>': 0}
        self.vectors = [None]
        
        print 'Start loading word vectors...'
        with open(path, 'r') as fp:
            for line in fp:
                segs = line.split()
                word = segs[0]
                vector = np.array(map(float, segs[1:]))
                self.indices[word] = len(self.indices)
                self.vectors.append(vector)
                if len(self.vectors) % 100000 == 0:
                    print 'Parsed %d lines.' % len(self.vectors)
                    
        self.vectors[0] = np.zeros_like(self.vectors[1])        
        print ('Completed! Established a dictionary with size: %d' % 
            len(self.vectors))
    
    def index_of(self, word, lower=True):
        """
        """
        
        return self.indices.get(word.lower(), len(self.vectors) - 1)
    
    def to_array(self):
        """
        """
        
        return np.vstack(self.vectors)

def make_data_frame(source, dest):
    with open(source) as fp:
        df = json_normalize(map(json.loads, fp))
    
    def vectorizer(text):
        if not isinstance(text, str):
            return None
        tokens = word_tokenize(text.lower())
        tokens = filter(lambda t: len(t) > 1, tokens)
        indices = map(dictionary.index_of, tokens)
        valid_indices = filter(lambda i: i != -1, indices)
        if len(valid_indices) < len(indices) * 0.8:
            return None
        return np.array(valid_indices)
        
    df['vector'] = df['text'].apply(vectorizer)
    pd.to_pickle(df, dest)
    
    
if __name__ == '__main__':
    dictionary = Dictionary(glove100d_path)
    make_data_frame(data_path, dest_path)