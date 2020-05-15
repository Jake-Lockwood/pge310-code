#!/usr/bin/env python
# coding: utf-8
# %%

# # Assignment 13
# 
# I have provided Python/NumPy implementations of both the [Gaussian elimination](https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_DirectSolvers.html#Python/NumPy-implementation-for-Gaussian-elimination-with-back-substitution-and-partial-pivoting) and [Gauss-Jordan elimination](https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_DirectSolvers.html#Python/NumPy-implementation-for-Gauss-Jordan-elimination-with-partial-pivoting) in the course notes.  Because these functions make heavy use of NumPy broadcasting, they are about as fast as they can be in Python, but this efficiency is at the expense of code readability.  Additionally, the way they are coded violates the ["Single-responsibility principle"](https://en.wikipedia.org/wiki/Single-responsibility_principle) of programming, i.e. each function should have a single well-defined task.
# 
# We can make these functions more readable and maintainable by using an object-oriented approach.  Your assignment is to complete the `LinearSystem` class below.  Specifically, you can separate the Gaussian elimination code into a function called `row_echelon` that puts the augmented system matrix into row echelon form, and `back_substitute` that performs that back substitution task.  These functions can then be called in sequence to solve a linear system of equations as shown in `gauss_solve`.  You can check your implementation by comparing with the output of [`numpy.linalg.solve`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html) function.
# 
# Additionally, complete the function `reduced_row_echelon`.  This function should perform Gauss-Jordan elimination to put the augmented system matrix in reduced row echelon form.  This can then be used to solve for the inverse of a matrix as implemented in the `inverse` function.  You can check your implementation by comparing with the output of [`numpy.linalg.inv`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html).
# 
# The class `LinearSystem` can be instantiated with a matrix `A` and right-hand side vector `b` for the solution of linear systems, or with only a matrix `A` for inverse computations.  The class should inherit from the `Matrix` class you implemented in [assignment12](https://github.com/PGE310-Students/assignment12) and use the member functions `row_swap`, `row_multiply` and `row_combine` to perform the matrix manipulations.
# 
# **Tip:** A simple way to turn your Jupyter notebook from [assignment12](https://github.com/PGE310-Students/assignment12) into an executable and importable Python module is to run the following command on the Terminal command line:
# 
# ```bash
# jupyter nbconvert --to python assignment12.ipynb
# ```
# 
# This will turn `assignment12.ipynb` into `assignment12.py` which should contain the `Matrix` class.  Then, `cp` or `mv` `assignment12.py` into this repository (i.e. [assignment13](https://github.com/PGE310-Students/assignment13)).  Now you should be able to add
# 
# ```python
# from assignment12 import Matrix
# ```
# 
# to your code and have the `Matrix` class available to inherit from in `LinearSystem`.  Don't forget to run `git add assignment12.py` when committing your final solution if you do this.  The alternative is to copy the `Matrix` class code into this assignment.

# %%


import numpy as np
from assignment12 import Matrix
class LinearSystem(Matrix):
    
    def __init__(self, A, b=None):
        
        self.mat = np.array(A, np.double)
         
        if b is not None:
            self.mat = np.c_[self.mat, b]
            
    def row_echelon(self):
    
    #Get the number of rows
        N = self.mat.shape[0]
    
    #Loop over rows
        for i in range(N):
            
        #Find the pivot index by looking down the ith 
        #column from the ith row to find the maximum 
        #(in magnitude) entry.
            p = np.abs(self.mat[i:, i]).argmax()
            
        #We have to reindex the pivot index to be the 
        #appropriate entry in the entire matrix, not 
        #just from the ith row down.
            p += i 
    
        #Swapping rows to make the maximal entry the 
        #pivot (if needed).
            if p != i:
                self.row_swap(i, p)
                
            for j in range(i+1, N):
                #Eliminate all entries below the pivot
                factor = self.mat[j, i] / self.mat[i, i]
                self.row_combine(j, i, factor)
        
        return #The function should manipulate self.mat
               #in place. Do not return anything.
                
    def reduced_row_echelon(self):
    
    
    #Get the number of rows
        N = self.mat.shape[0]
    
    #Loop over rows
        for i in range(N):
            
        #Find the pivot index by looking down the ith 
        #column from the ith row to find the maximum 
        #(in magnitude) entry.
            p = np.abs(self.mat[i:, i]).argmax()
            
        #We have to reindex the pivot index to be the 
        #appropriate entry in the entire matrix, not 
        #just from the ith row down.
            p += i 
    
        #Swapping rows to make the maximal entry the 
        #pivot (if needed).
            if p != i:
                self.row_swap(i, p)              

            
            for j in range(N):
                if i!=j:
                    
                    #Eliminate all entries above and below the pivot
                    factor = self.mat[j, i] / self.mat[i, i]
                    self.row_combine(j, i, factor)
                    
            #Make the diagonals one
            self.row_multiply(i, 1/self.mat[i, i])
            
    def back_substitute(self):
        
        (N, M) = self.mat.shape
        
        #Allocating space for the solution vector
        x = np.zeros(N, dtype=np.double)

        #Here we perform the back-substitution.  Initializing 
        #with the last row.
        x[-1] = self.mat[-1,-1] / self.mat[-1, -2]
    
        #Looping over rows in reverse (from the bottom up), starting with the second to
        #last row, because the last row solve was completed in the last step.
        for i in range(N-2, -1, -1):
            x[i] = (self.mat[i,-1] - np.dot(self.mat[i,i:-1], x[i:])) / self.mat[i,i]
        
        return x #Return the solution vector.
    
    def gauss_solve(self):
        ##########################
        ##### Do not change ######
        ##########################}
        self.row_echelon()
        return self.back_substitute()
    
    def inverse(self):
        ##########################
        ##### Do not change ######
        ##########################
        N = self.mat.shape[0]
        self.mat = np.c_[self.mat[:,:N], np.eye(N)]
        self.reduced_row_echelon()
        return self.mat[:,-N:]


# %%


#ls = LinearSystem([[1,3,4], [5, 4, 2], [1, 7, 9]], b=[[1], [1], [1]])
#ls1 = LinearSystem([[1,2,3], [2,-1,1], [3,0,-1]], b=[[9], [8], [3]])
#ls2 = LinearSystem([[1,1,1], [1,-1,-1], [0,1,1]], b=[[6], [-4], [-1]])
#ls3 = LinearSystem([[1,1,-1], [2,0,1], [0,1,1]], b=[[0], [14], [13]])
#ls4 = LinearSystem([[1, 2, -1, 1], [-1, 1, 2, -1], [2, -1, 2, 2], [1, 1, -1, 2]], b=[[6], [3], [14], [8]])
#ls3.reduced_row_echelon()
#ls4.reduced_row_echelon()
#ls4.gauss_solve
#print(ls4)
#ls1.row_echelon()
#ls3.back_substitute
#print(ls3)
#print(ls1)
#ls1.back_substitute()


# %%





# %%




