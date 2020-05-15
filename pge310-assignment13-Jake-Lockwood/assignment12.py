#!/usr/bin/env python
# coding: utf-8

# # Assignment 12
# 
# Complete the `Matrix` class below by implementing the three member functions that perform the following matrix row operations:
# 
#  * `row_swap`: Should swap rows `i` and `j` in a matrix.
#  
#  * `row_multiply`: Should multiply row `i` by `factor`.
#  
#  * `row_combine`: Should perform the operation
#  
#     $$
#     A_i = A_i - m  A_j
#     $$
#     
#     where $A$ is a matrix and $i$ and $j$ are rows of the matrix.  $m$ is a     scalar multiplication `factor`.
#     
# The operations should perform the matrix manipulations to the `mat` class attribute *in place*, i.e. they should **not** return a new matrix.  You can assume that `mat` will always be a NumPy array.

# In[4]:


import numpy as np

class Matrix(object):
    
    def __init__(self, array):
        
        if not isinstance(array, (list, np.ndarray)):
            raise TypeError('Input matrix must be a Python list or NumPy ndarray.')
        else:
            self.mat = np.array(array)
        
    def __str__(self):
        return str(self.mat)
    
    def row_swap(self, i, j):
        
        #row_1 = self.mat[i]
        #row_2 = self.mat[j]
        #self.mat.put(self.mat, [i, j], [j, i])
        self.mat[[i, j]] = self.mat[[j, i]]
        
        return
    
    def row_multiply(self, i, factor):
        
        #self.mat.put(self.mat, [i], [i*factor])
        self.mat[[i]] = self.mat[[i]]*factor
    
        return
        
    def row_combine(self, i, j, factor):
        #multiply = self.mat[j*factor]
        #new = np.subtract(self.mat[i], multiply)
        
        self.mat[[i]] = self.mat[[i]]-self.mat[[j]]*factor
        #self.mat.put(self.mat, [i], [i-factor*j])
        
        return


# In[ ]:


#b = np.array([1, 2, 3, 4]).np.arrange(2, 2)
#a = Matrix(b)
#print(b)


# In[ ]:




