These are sample outputs of the program trainPyLearn.py

The grid code is generated with default parameters 
lambda = (9, 10, 11, 13), and rho = 0. 5

The cliques are random cliques on 7 vertices, in a graph of total 14 vertices. 

For convenience, we represent a clique as a binary vector of length 14, 1 at 
location i if the ith vertex is in the clique. 

Since we only have a feed forward network (no hidden layer), training on the entire network
is equivalent to training at one vertex at a time. 

The program pairs the good code words (those in the range [0, R_\ell]) to a randomly chosen clique. 
Then with the code words as inputs, for each i in 0 to 13, 
train a feed forward neural network with the indicator of the i-th vertex as the output node. 

Here are the file descriptions:
- targets.csv : list of randomly chosen cliques to be learned
- input.csv: input file for the learning problem, in the notation of pylearn2.
The first column is the binary output variable (1 if vertex in clique, 0 else).
The next 4 columns form the vector (x mod 9, x mod 10, x mod 11, x mod 13), 
which is our codeword.
- output.csv: prediction of the models on the training input. 






