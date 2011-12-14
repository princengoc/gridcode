'''Implement the sparse hopfield 3-parameter model. 
Input: a graph as its edge vector
Output: the cleanup-ed graph. 
Class: CliqueFinder(K, V)
Internally store: x,y,z
Internal functions: 
heaviside
updateEdge
updateOneStep (update one step at a time)
updateGraph: update all steps
'''

import itertools
import numpy as np
import scipy.sparse as sparse

def updateEdge(param,  edge,  vec,  V):
    '''Function: quick calculation of the update function: (J*x)_{e} - J_{ee}
    This function takes in an array of parameters (x,y,z), 
    an edge e as a set, a binary vector vec \in {0,1}^{binom(V,2)} 
    V (the number of nodes), and returns
    the number Jx_{e} - J_{ee}, to be used in updating the value of x_e in a Hopfield network:
    x_e^{new} = 1 if (J*x)_{e} - J_{ee} > 0, = 0 else. '''
    '''param is a numpy array, edge is a set, and vec is a binary 0/1 SPARSE COLUMN VECTOR of length n = binom(V,2)'''
    #NOTE: can be optimized further since x,y,z this case are rationals.
    #for the moment, store x,y,z as 64 float.
    x = param[0]
    y = param[1]
    z = param[2]
    i = edge.pop()
    j = edge.pop()
    #reorder to make sure i > j
    if(i < j):
        temp = i
        i = j
        j = temp
    #create the vector of non-zero locations
    row = np.array(range(i) + [i]*(V-i-1) + range(j) + [j]*(V-j-2))
    col = np.array([i]*i + range(i+1,  V) + [j]*j + range(j+1,  i) + range(i+1, V))
    loc = row*V - row*(row+1)/2 + col - (row+1)
    #now create a ROW binom(v,2) \times 1 sparse matrix, with non-zero values x-y
    data = [x-y]*len(loc)
    #guys corresponding to the diagonal should get -y
    #by the requirement i > j, this is always the j position on the row and col
    data[j] = -y
    
    #concatenate into one data file
    zer = np.array([0]*len(loc)) #a zero list of the correct dimension, to make the sparse vector
    Jrow = sparse.csr_matrix((data,  (zer,  loc)),  shape = (1,  V*(V-1)/2))    

    #multiply to get (Jvec)_e - z = [x-y]*vec + y*sum(vec) - z
    val = (Jrow*vec).toarray() + y * sum(vec.data) - z #TODO: speed up this computation
    return val.item()

def bn(ij,  V):
    """ Converts an edge (i,j) to an index. NOTE: ij is just a normal list or array, NOTE a set
    """
    i = ij[0]
    j = ij[1]
    if (i < j):
        return i*V - i*(i+1)/2 + j-(i+1)
    else:
        return j*V - j*(j+1)/2 + i-(j+1)

#update a graph ONE STEP
#graph sigma is stored as V*(V-1)/2 vector in sparse form

def heaviside(val):
    '''Return 1 if val >= 0, 0 else '''
    if(val >= 0):
        return 1
    else:
        return 0

#NOTE: sigma is stored in lil format to make it easier to change the sparsity stucture
''' function: updateGraph: update until convergence'''

def updateGraph(param, sigma, V):
    pairs = itertools.combinations(range(V),  2) #fix the update order to be the same as that in Chris'
    change = False
    while(change != True):
        change = True
        #pairs = np.random.permutation(list(pairs)) #DO A RANDOM UPDATE ORDER IN EACH ITERATION
        for item in pairs:
            edge = set(item)
            #store old value
            oldval = sigma[bn(item,  V),  0]
            newval = heaviside(updateEdge(param,  edge,  sigma.tocsc(),  V))
            if (oldval != newval):
                sigma[bn(item,  V),  0] = newval
                change = True
                #print "value updated at edge %s to %d" % (str(item), newval)
    return sigma



'''Function: take a vertex list and returns a ER(V, p) with the clique being the vertex list'''
def noisyClique(p, V, vertexList, random = True): #TODO: speed up this step. 
    if (random):
        updateNum = np.random.binomial(V*(V-1)/2, p) #number of guys shall be updated
    else:
        updateNum = p    #else: p is the number of bit flips. 
    updateIndex = np.random.permutation(range(V*(V-1)/2))[0:updateNum] #choose noisy edges
    #make the clique
    pairs = np.sort(list(itertools.combinations(vertexList,  2)), 1).T
    row = pairs[0]
    col = pairs[1]
    loc = row*V - row*(row+1)/2 + col - (row+1)
    #remove redundancies
    loc = np.concatenate((loc, updateIndex))
    set = {}
    map(set.__setitem__, loc, [])
    loc = set.keys()
    #generate the corresponding sparse vector
    data = [1]*len(loc)
    zer = [0]*len(loc) #a zero list of the correct dimension, to make the sparse vector
    cliqueAndNoise = (sparse.csc_matrix((data,  (loc, zer)),  shape = (V*(V-1)/2, 1),  dtype = 'int8')).tolil()
    return cliqueAndNoise

''' Function: from a graph described by its edge vector, find the vertex with the maximum degree'''
def vertexMaxDegree(V, sigma):
    #dumb code: make a subsparse matrix and ask for the sum
    degarray = []
    maxindex = []
    for i in xrange(V):
        row = np.array(range(i) + [i]*(V-i-1))
        col = np.array([i]*i + range(i+1, V))
        loc = row*V - row*(row+1)/2 + col - (row+1)
        ideg = sum(sigma[loc,0].toarray())[0]
        degarray = degarray + [ideg]
    #reiterate again and check who attained the maximum
    maxdeg = max(degarray)
    for i in xrange(V):
        if degarray[i] == maxdeg:
            maxindex = maxindex + [i]
    #return the whole list of vertex of maxdegree
    return maxindex

'''Function: generate the true clique in a sparse representation style
Input: vertices want to include in the clique
Output: the vector in sparse representation'''
def sparseKClique(V, vertexList):
    pairs = np.sort(list(itertools.combinations(vertexList,  2)), 1).T
    row = pairs[0]
    col = pairs[1]
    loc = row*V - row*(row+1)/2 + col - (row+1)    
    #generate the corresponding sparse vector
    data = [1]*len(loc)
    zer = np.array([0]*len(loc)) #a zero list of the correct dimension, to make the sparse vector
    clique = sparse.csc_matrix((data,  (loc, zer)),  shape = (V*(V-1)/2, 1))    
    return cliquem
