# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 23:06:41 2014

@author: princengoc

Approximately, a grid cell code stores a number in base (\lambda_1, ... \lambda_N).
Such a code can store numbers in [0, R]
for R = \lambda_1 \times ... \times \lambda_N. The desired code words are numbers
in [0, r] for some r < R. We want a Hopfield decoder. 

Given a number x in [0, R], call (x mod lambda_1, ... x mod lambda_M) its
word representation. The binary form of a word is a vector of length R, 
which comes in M groups, where in the j-th group, the i-th entry is 1 represents
i mod lambda_j.

A three-parameter clique network has (v \choose 2) neurons, each indicates the
presence of an edge in an undirect graph on v nodes. A code word corresponds to a
simple graph. The desired code words are cliques of size v/2 (or cliques of
size k in some appropriate range, in general). 

This program trains a single layer neural network (SLN) to map desired code words
to k-cliques. The training algorithm is perceptron (delta rule), implemented in
the package neurolab http://code.google.com/p/neurolab/

Given a perturbed grid cell code word w' within a small \ell_2 distance to w,
let SLN(w') be the input to the Hopfield network, and Hopf(SLN(w')) be 
the output of the Hopfield dynamic with the given input. 

Ideally, we want Hopf(SLN(w')) = Hopf(w) = clique. This means the Hopfield
network can be used for decoding the grid cell code.

"""

import numpy as np
#lam = (15, 17, 19, 23)

import operator as op
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

class Gridcode:
    def __init__(self, lam, rho):
        """ Create a grid cell code.
        lam: phase vector
        rho: r = R^rho, where r is the range of the desired code"""
        self.lam = lam
        self.R = reduce(lambda x,y: x*y, lam)
        self.N = len(lam)
        self.D = reduce(lambda x,y: x+y, lam)
        self.r = int(self.R**rho)
        self.goodWord = {}
        self.goodBin = {}
        self.goodZ = {}
        
    def numToWord(self, num):
        """Given a number num in [0,R], returns the word representation
        in lambda basis"""
        return [num % ell for ell in self.lam]
       
    def wordToBinary(self, word):
        """Given a word in lambda basis, returns the binary representation"""
        x = []
        for i in range(self.N):
            xi = [0]*self.lam[i]
            xi[word[i]] = 1
            x.append(xi)
        return x
       
    def neighborhood(self, word, dmax = 1):
        """Returns a list of words differ from word by 
        exactly dmax in one of the coordinates.
        """
        neighbor = []
        for i in self.N:
            wi = word + []
            wi[i] = (word[i] + 1) % self.lam[i]
            wi2 = word + []
            wi2[i] = (word[i] - 1) % self.lam[i]
            neighbor.append(wi)
            neighbor.append(wi2)
        return neighbor

    def computeWordDict(self):
        """Computes self.goodWords, a dictionary mapping integers in [0,r] to words."""
        if(len(self.goodWord) == 0):
            for num in xrange(self.r):
                self.goodWord[num] = self.numToWord(num)
    
    def computeBinaryDict(self):
        """Computes self.goodBin, a dictionary mapping integers in [0,r] to words,
        under binary representation"""
        if(len(self.goodBin) == 0):
            for num in xrange(self.r):
                self.goodBin[num] = self.wordToBinary(self.numToWord(num))      

    def getInput(self):
        """Returns the input vector for the single layer neural network training"""
        return [reduce(lambda s,t: s + t, x) for x in gc.goodBin.values()]
        
    def wordToNum(self, word):
        """Given a word, returns a number in [0,R]. 
        Algorithm: starts with the largest prime lam_m, and search over numbers
        of the form word_m + i*lam_m for one with correct modulo in lam_{m-1}.
        Once we found a candidate (eg: x), search over numbers of the form
        x + i*lam_m*lam_{m-1} to match word_{m-2}, and so on."""
        pass

    def project(self, word): 
        """Take a word in [0, R], returns the nearest code word
        in the range [0,r]. 
        Comment: this seems to be a hard problem."""        
        pass

    
