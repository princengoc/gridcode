Problem background: 

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

Given a perturbed grid cell code word w' within a small \ell_2 distance to w,
let SLN(w') be the input to the Hopfield network, and Hopf(SLN(w')) be 
the output of the Hopfield dynamic with the given input. 

Ideally, we want Hopf(SLN(w')) = Hopf(w) = clique. This means the Hopfield
network can be used for decoding the grid cell code.

-------------------
What this program does:

This program trains a single layer neural network (SLN) to map desired code words
to k-cliques. The training algorithm is perceptron (delta rule), implemented in pylearn2. 

More info about pylearn2:
http://deeplearning.net/software/pylearn2/

Handy tutorials.
http://fastml.com/pylearn2-in-practice/

We use yaml input to define and train our feed forward network, as explained in
http://nbviewer.ipython.org/github/lisa-lab/pylearn2/blob/master/pylearn2/scripts/tutorials/softmax_regression/softmax_regression.ipynb

The main file is in trainPyLearn.py
The folder output contains some sample outputs. 
-------------------



