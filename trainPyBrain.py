# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:55:44 2014

@author: princengoc

Training using pybrain
"""
from gridcells import Gridcode, ncr

gc = Gridcode((9, 10, 11, 13), 0.5)
gc.computeBinaryDict()
input = gc.getInput()

#take some collection of cliques as targets
from cliqueFinder import randomClique
V = 16
targets = []
vertexList = set()
ctr = 0
while ctr < gc.r:
    clique, vx = randomClique(V)
    if(vx not in vertexList):
        vertexList.add(vx)
        targets.append(clique.tolist())
        ctr = ctr + 1


#using pybrain
#test their xor example
#net = buildNetwork(2, 3, 1, bias = 'True')
if(False):
    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.datasets import SupervisedDataSet
    from pybrain.supervised.trainers import BackpropTrainer
    from pybrain.structure.modules import TanhLayer
    from pybrain.structure import FeedForwardNetwork    
    from pybrain.structure.modules import *    
    net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer)
    ds = SupervisedDataSet(2, 1)
    ds.addSample((0, 0), (0,))
    ds.addSample((0, 1), (1,))
    ds.addSample((1, 0), (1,))
    ds.addSample((1, 1), (0,))
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence()
   
    net = buildNetwork(gc.D,ncr(V,2), bias = True)
    ds = SupervisedDataSet(gc.D, ncr(V,2))
    for i in xrange(gc.r):
        ds.addSample(input[i], targets[i])
        
    trainer = BackpropTrainer(net, ds)
    trainer.train()
    trainer.trainUntilConvergence()
    #--- Error (mean squared error?): 0.16/2 = 0.08. 
    [x > 0.5 for x in net.activate(input[1])]
    targets[1]
