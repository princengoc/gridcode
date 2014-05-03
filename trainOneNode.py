# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 15:49:59 2014

@author: princengoc

Trying using pylearn2 to learn a one vertex.
"""

def softMaxFit(i):
    """Fit a feed forward neural network, with data input
    data(i).csv"""
    import numpy as np    
    datapath = '/home/princengoc/hopfield/data'+str(i)+'.csv'
    
    with open('sr_dataset.yaml', 'r') as f:
        dataset = f.read()
    
    dataset = dataset %locals()
    print dataset
    
    with open('sr_model.yaml', 'r') as f:
        model = f.read()
    f.close()
    print model
    
    with open('sr_algorithm.yaml', 'r') as f:
        algorithm = f.read()
    f.close()
    
    print algorithm
    
    with open('sr_train.yaml', 'r') as f:
        train = f.read()
    save_path = '.'
    
    train = train %locals()
    
    from pylearn2.config import yaml_parse
    train = yaml_parse.load(train)
    train.main_loop()
    
    #save predictions
    from pylearn2.utils import serial
    import theano
    model_path = 'softmax_regression.pkl'
    model = serial.load(model_path)
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    #Y = theano.tensor.argmax(Y, axis=1)
    f = theano.function( [X], Y )
    fname = 'input' + str(i) + '.csv'
    testx = np.loadtxt(fname, delimiter = ',')
    y = f(testx)
    fname = 'output' + str(i) + '.csv'    
    np.savetxt(fname,y[:,1],delimiter = ',')