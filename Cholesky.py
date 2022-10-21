#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
from scipy.sparse import diags


# In[35]:


# Program is solving  system of equations(tridiagonal matrix + numbers in corners)
def cholesky(v1, v2, b, x):
    y = np.zeros(7)
    Chol1 = np.zeros(7)
    Chol2 = np.zeros(7)
    
    Chol1[0] = math.sqrt(v2[0])
    for i in range (1,7):
        Chol2[i-1] = v1[i-1]/Chol1[i-1]
        Chol1[i] = math.sqrt(v2[i] - Chol2[i-1] * Chol2[i -1])
    
    y[0] = b[0] / Chol1[0]
    for i in range(1,7):
        y[i] = (b[i] - y[i-1] * Chol2[i-1]) / Chol1[i]
    
    x[6] = y[6] / Chol1[6]

    for i in range(5, -1, -1):
        x[i] = (y[i] - Chol2[i] * x[i+1]) / Chol1[i]
    


# In[8]:


v_ad = [1] * 7
v_d = np.array([3,4,4,4,4,4,3])
v_ex = np.array([1,0,0,0,0,0,1])
v_resultsb = np.array([1,2,3,4,5,6,7])
v_z = [0] * 7
v_q = [0] * 7
v_resultx = [0] * 7


# In[37]:


cholesky(v_ad, v_d, v_resultsb, v_z) #A*z=b
cholesky(v_ad, v_d, v_ex, v_q) #A*q=u

v = (v_z[0] + v_z[6]) / (1 + v_q[0] + v_q[6])


# In[40]:


for i in range(0,7):
    v_resultx[i] = v_z[i] - (v * v_q[i])
print(v_resultx)


# In[ ]:




