# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:24:09 2014

@author: tran

This program solves a system of linear inequalities to 
check if the Hamming(7,4) code
can be decoded by a Hopfield network on 7 nodes.
(And similarly for the Hamming 84 code)
"""

from cvxopt import matrix, solvers
import numpy as np
import operator

class Hamming74:
    """Manages the Hamming 74 code. 
    First 4 digits are data, last 3 are parity."""
    def __init__(self):
        self.code = []
    
    def appendParity(self, d):
        """Given data, a binary vector of length 4,
        returns the coded vector of length 7"""
        p01 = (d[0]+d[1]+d[3]) % 2
        p02 = (d[0]+d[2]+d[3]) % 2
        p12 = (d[1]+d[2]+d[3]) % 2
        return d + [p01, p02, p12]
    
    def computeState(self):
        """Compute all 2**4 = 16 code words"""
        if(len(self.code) == 0):
            for i in [0,1]:
                for j in [0,1]:
                    for k in [0,1]:
                        for l in [0,1]:
                            d = [i,j,k,l]
                            w = self.appendParity(d)
                            self.code = self.code + [w]

class Hamming84(Hamming74):
    """Manages the Hamming 84 code.
    First 4 digits are data, last 4 are parity."""
    def updateState(self):
        self.computeState()
        for i in xrange(len(self.code)):
            w = self.code[i]
            self.code[i] = w + [reduce(operator.add,w) % 2]
            

class HopfieldSolver:
    """Given a set of codewords, solves a linear program
    to determine if there exists a Hopfield network
    that stores these codewords as 1-stable points. 
    Assume n \times n symmetry matrix J. This creates a LP of the form:
    min 0\cdot xs
    st. Gx + s = 0
    s >= 0
    x is vectorized J, so it has dimension N= n(n+1)/2
    G has dimension M \times N, where M = # codewords * n
    (since each codeword can have n bit flips, each flip
    creates one constraint.)
    """
    def __init__(self, codes, n):
        self.codewords = codes
        self.n = n
        self.N = int(n*(n+1)/2)
        self.codeWell = {}
        self.G = []
        #names of columns of G as indexed by entries of J
        self._gnames = []
        #dictionary mapping i to indices of gnames containing i
        self._nameloc = {} 

    @property
    def gnames(self):
        """Rows of G are indexed lexicographically, 
        gnames stores these names as pairs. 
        00,01,...0(n-1),11,12,...,(n-1)(n-1)"""
        if(len(self._gnames) == 0): 
            for i in range(self.n):
                for j in range(i, self.n):
                    self._gnames = self._gnames + [(i,j)]
        return self._gnames

    @property
    def nameloc(self):
        if(len(self._nameloc) == 0):
            for i in range(self.n):
                self._nameloc[i] = self.__nameloc(i)        
        return self._nameloc
    
    def corrupt(self, word):
        """Given a word, returns the list of n
        possible one-bit corruptions of this word"""
        pass
    
    def __nameloc(self,i):
        """Given i in range(n), 
        returns indices in range(N) of the form ij or ji"""
        return tuple([l for l in xrange(self.N) if(i in self.gnames[l])])
    
    def flipi(self, word, i):
        """Given a word, returns the inequality vector of G
        correspond to requiring that the i-th bit corruption of
        word has bigger energy. """
        sgn = 1-2*word[i]
        row = [0]*self.N
        nl = self.nameloc[i]
        for j in range(self.n):
            if(j != i):
                row[nl[j]] = word[j]*sgn
            else:
                row[nl[j]] = sgn
        return row
            
    def flip(self,word):
        """Given a word, returns the list of 
        n one-bit flips inequalities"""
        mat = []
        for i in xrange(self.n):
            mat = mat + [self.flipi(word,i)]
        #return np.vstack(mat*1.0)
        return mat
    
    def genG(self):
        """Generate G, the  M \times N numpy matrix"""
        if(len(self.G) == 0):
            for word in self.codewords:
                self.G = self.G + self.flip(word)
            self.G = np.vstack(self.G)*1.0
        return self.G
    
    def canDecode(self):
        """Solve the linear program Gx < 0. 
        If infeasible, this means there is no Hopfield network
        that can cope with one-bit corruption of the specified
        codewords. 
        If feasible, this means there is such a Hopfield network. 
        The x vector in the solution to the linear program would 
        give one such network"""
        self.genG()
        G = matrix(self.G)
        c = matrix([0.0]*shape(G)[1])
        h = matrix([-1.0]*shape(G)[0])
        sol = solvers.lp(c,G,h)
        if sol['status'] == 'optimal':
            return True
        if 'infeasible' in sol['status']:
            return False
        
    
#check feasibility for the Hamming 7,4 code
hm = Hamming74()
hm.computeState()
hs = HopfieldSolver(hm.code, 7)
hs.canDecode()
#Infeasible --> no Hopfield network can do 1-bit corruption decoding.

#try Hamming 8,4 code
hm = Hamming84()
hm.updateState()
hm.computeState()
hs = HopfieldSolver(hm.code, 7)
hs.canDecode()
