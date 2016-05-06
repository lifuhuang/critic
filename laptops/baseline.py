import sys
import os.path as op
import numpy as np

######
sys.path.append('/home/lifu/PySpace/dimsum')
######

from dimsum import activations
from dimsum import optimizers
from dimsum import objectives
from dimsum.layers import Dense, Input
from dimsum.models import NeuralNetwork
from dimsum.callbacks import LossPrinter, IterationPrinter

shared_path = '/mnt/shared/'
glove100d_path = '/mnt/shared/glove/glove.6B.100d.txt'
corpus_path = '/mnt/shared/beeradvocate.txt'
params_path = 'params.npz'

def main():  
    X = np.load('/mnt/shared/beeradvocate/X_1.npy')
    Y = np.load('/mnt/shared/beeradvocate/Y_1.npy')    
    print 'Data loaded! %d samples in total.' % X.shape[0]

    model = NeuralNetwork(objective=objectives.MeanSquareError)
    model.add(Input(100))
    model.add(Dense(100, 512, activation=activations.ReLU))
    model.add(Dense(512, 256, activation=activations.ReLU))
    model.add(Dense(256, 128, activation=activations.ReLU))
    model.add(Dense(128, 5))

    if op.exists(params_path):
        print 'Found saved params, loading params...',
        model.load_params(params_path)
        print 'Done!'
    
     
    print 'Model constructed!'
    
    split = X.shape[0] // 10 * 9
    
    X_train = X[:split]
    Y_train = Y[:split]
    X_dev = X[split:]
    Y_dev = Y[split:]
    
    callbacks = [(IterationPrinter(sys.stdout), 100),
                 (LossPrinter(outfd=sys.stdout, update_rate=0.1), 100), 
                 (LossPrinter(x=X_dev, y=Y_dev, outfd=sys.stdout), 5000)]
    try:
        model.fit(X_train, Y_train, n_epochs=10, batch_size=64, 
                  optimizer=optimizers.Adagrad(0.3), callbacks=callbacks)
    except KeyboardInterrupt:
        print 'Terminated!'
    finally:
        model.save_params(params_path)
        print 'params saved'
        
    yh = model.predict(X_dev[:10])
    
    for i in xrange(10):
        print '%d:' % i
        print 'Y_true', Y_dev[i]
        print 'Y_pred', yh[i]

if __name__ == '__main__':
    main()