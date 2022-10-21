#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA


# In[2]:


# Problem is to find eigenvalue with the highest absolute value and corresponding to them eigenvectors
def eigenvalue1(A):
    y_prev = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    while True:
        zk = np.matmul(A, y_prev)
        norm_zk = LA.norm(zk)
        y_next = [zk[i]/norm_zk for i in range(0,len(zk))]
        if abs(abs(y_prev[0]) - abs(y_next[0])) < 1e-8:
            print(f" wektor wlasny: {y_next}")
            print(f" wartosc wlasna: {norm_zk}")
            return y_next
        y_prev = y_next


# In[3]:


def orthogonalization(zk, e):
    x = 0
    for i in range(0,len(zk)): 
        x = x + zk[i] * e[i]
    for i in range(0,len(zk)):
        zk[i] = zk[i] - e[i] * x
    return zk
    


# In[4]:


def eigenvalue2(A, eigenvalue):
    y_prev = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    while True:
        zk = np.matmul(A, y_prev)
        zk = orthogonalization(zk, eigenvalue)
        norm_zk = LA.norm(zk)
        y_next = [zk[i]/norm_zk for i in range(0,len(zk))]
        if abs(abs(y_prev[0]) - abs(y_next[0])) < 1e-8:
            print(f" wektor wlasny: {y_next}")
            print(f" wartosc wlasna: {norm_zk}")
            return y_next
        y_prev = y_next


# In[5]:


A = np.array([
    [19/12, 13/12, 5/6, 5/6, 13/12, -17/12],
    [13/12, 13/12, 5/6, 5/6, -11/12, 13/12],
    [5/6, 5/6, 5/6, -1/6, 5/6, 5/6],
    [5/6, 5/6, -1/6, 5/6, 5/6, 5/6],
    [13/12, -11/12, 5/6, 5/6, 13/12, 13/12],
    [-17/12, 13/12, 5/6, 5/6, 13/12, 19/12],
], np.float64)
#print(A)
print(type(A))
eigenvalue_1 = eigenvalue1(A)
eigenvalue_2 = eigenvalue2(A, eigenvalue_1)
#n = np.linalg.eig(A)
#print(n)


# In[24]:





# In[25]:





# In[ ]:




