# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:24:09 2014

@author: tran

This code solves a linear inequality to 
check if the Hamming(4,7) code
can be decoded by a Hopfield network on 7 nodes.
"""

class Hamming47:
    """Manages the Hamming 47 code"""
    def __init__(self):
        self.codes = {}
    
    def computeState(self):
        """Compute all 2**4 = 16 code words"""
        pass

    def getParity(self, data):
        """Given data, a binary vector of length 4,
        returns the there parity bits"""
        pass
    
    def corrupt(self, code):
        """Given a codeword code, return the list of 7
        possible one-bit corruptions of this code"""
        pass
    
#generate the inequalities for the J matrix
#form the giant matrix inequality Ax < 0
#use LP to solve for Ax < 0. 