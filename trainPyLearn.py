# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 12:56:51 2014

@author: princengoc

Train using pylearn2. We shall use the word representation. 
Also, we represent a clique by its set of vertices, rather than its
set of edges.
"""

from gridcells import Gridcode, ncr
from trainOneNode import softMaxFit
import numpy as np

gc = Gridcode((9, 10, 11, 13), 0.5)
gc.computeWordDict()

#take some collection of cliques as targets
from cliqueFinder import randomVertex
V = 14
targets = []
vertexList = []
ctr = 0
vxlist = None
while ctr < gc.r:
    vxlist, vxvec = randomVertex(V, frac = 0.5, vxlist = vxlist)
    if(vxvec not in targets):
        targets.append(vxvec)
        vertexList.append(vxlist)
        ctr = ctr + 1

#sort the targets
dtype = []
for i in range(int(V/2)):
    dtype = dtype + [(str(i), int)]
vertexList = np.sort(vertexList, 1)


#save the targets as well
import csv
fname = 'targets.csv'
with open(fname, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    for j in xrange(gc.r):
        writer.writerow(targets[j])

#save to csv. Save one file for each vertex, so we can do binary classification
for i in xrange(V):
    #data in (y, x) format
    fname = 'data' + str(i) + '.csv'
    with open(fname, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for j in xrange(gc.r):    
            writer.writerow([targets[j][i]] + gc.goodWord[j])
    #data in (x) format, for prediction use
    fname = 'input' + str(i) + '.csv'
    with open(fname, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for j in xrange(gc.r):    
            writer.writerow(gc.goodWord[j])
    #fit model, save predictions into output(i).csv
    softMaxFit(i)

"""Analyze the output of trainPyLearn: for each input,
take the k vertices with the highest probability. 
Note that these probabilities may or may not be > 0.5"""
import numpy as np
i = 0
fname = 'output' + str(i) + '.csv'
ytab = np.loadtxt(fname, delimiter = ',')
ytab = ytab.reshape((gc.r, 1))

for i in range(1,V):
    fname = 'output' + str(i) + '.csv'
    ypred = np.loadtxt(fname, delimiter = ',')
    ypred = ypred.reshape((gc.r, 1))
    ytab = np.hstack((ypred, ytab))

#take the biggest k
k = int(V/2)
bigindex = np.argsort(ytab)[:,(k-1):(V-1)]
#compute % of vertices got correct
perc = []
for i in xrange(gc.r):
    numcor = len(set(vertexList[i]).intersection(set(bigindex[i])))
    perc = perc + [round(numcor/k, 1)]

    