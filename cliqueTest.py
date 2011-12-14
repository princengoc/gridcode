'''Implement the sparse hopfield 3-parameter model. 
Attempt to find cliques in a bunch of random graph with planted clique
'''
#from time import clock
import itertools
import numpy as np
import scipy.sparse as sparse
from cliqueFinder import *

#DEBUG RUN: 
#CREATE THE SAME STARTING GRAPH AND SEE WHAT THE TWO PROGRAMS DO
Vlist = range(100, 101)
ntrials = 100

perc = {}
percdumb = {}
for V in Vlist:    
    print "for V = %d we will count total number of times the max degree vertex of cleaned hidden clique graph intersects the actual hidden clique" % V
    tot = dict()
    totdumb = dict()
    kmin = 5
    kmax = 20
    #kmin = np.int(np.log(V))
    #kmax = np.int(np.sqrt(V))
    kd = 2
    for k in range(kmin,kmax,kd):
        tot[k] = 0 #counting SUCCESSES
        totdumb[k] = 0
        #set parameter
        param = np.array([(1.0 + (k-2)*(k-3))/(3*k - 8.0),  -1.,  -0.5])
        for trial in xrange(ntrials):
            idtrue = np.random.permutation(V)[:k] #pick k random vertices
            #generate the noisy clique
            noisyGraph = noisyClique(0.5, V, idtrue, random = True)
            print "trial %d " % trial
            print "number of edges added %d" % (noisyGraph.nnz - k*(k-1)/2)
            #immediately check the vertex of highest degree
            iddumb = set(vertexMaxDegree(V, noisyGraph))
            if (len(set(idtrue).intersection(iddumb)) != 0):
                totdumb[k] = totdumb[k] + 1
            #clean up
            sigma = updateGraph(param, noisyGraph, V)
            print "number ofedges after cleaning %d" %(noisyGraph.nnz - k*(k-1)/2)
            idx = set(vertexMaxDegree(V, sigma))
            #print guys
            print idx
            print idtrue
            if (len(set(idtrue).intersection(idx)) != 0): #if some of the max degree guy is in the clique
                tot[k] = tot[k] + 1
                print "total number of vertices with max degree = %d" % len(idx)
    print "finished for k = %d" % k
    perc[V] = [1.0*tot[k]/ntrials for k in xrange(kmin,kmax,kd)]
    percdumb[V] = [1.0*totdumb[k]/ntrials for k in xrange(kmin,kmax,kd)]
    

ks = range(kmin,kmax,kd)
import matplotlib.pyplot as plt
fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111)
for V in Vlist:
    ax.plot(ks,perc[V], label=str(V))
    ax.plot(ks,percdumb[V],  label = str("dumbV"))
plt.legend()
plt.show()


