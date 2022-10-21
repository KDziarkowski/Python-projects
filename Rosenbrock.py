#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install lmfit')


# In[2]:


# Program is looking for minimum of Rosenbrock function
from lmfit import minimize, Parameters, Parameter, report_fit


# In[6]:


#Rosenbrock function -> f(x, y) = (1 − x)^2 + 100(y-x^2)^2
import numpy as np
def fun_rosenbrock(params):
    return np.array([10 * (params['y'] - params['x']**2), (1 - params['x']), 2 * params['x']]) #returns the vector of residuals of the Rosenbrock function -> this is required by the minimize method


# In[7]:


def iter_cb(params, iter, resid):
    intermediate_values.append((params['x'].value, params['y'].value, calc_from_residuals(resid)))

def calc_from_residuals(residuals):
    return residuals[0]**2 + residuals[1]**2 + residuals[2]


# In[8]:


from random import randint
for i in range(10):
    params = Parameters()
    x0 = randint(0, 10)
    y0 = randint(0, 10)
    params.add('x', value=x0)
    params.add('y', value=y0)

    intermediate_values = []

    min = minimize(fun_rosenbrock, params, iter_cb=iter_cb)
    print(f'Starting points: ({x0}, {y0}) Number of steps: {min.nfev}')


# In[9]:


from scipy.optimize import  rosen

params = Parameters()
params.add('x', value=2.0)
params.add('y', value=2.0)

intermediate_values = []

min = minimize(fun_rosenbrock, params, iter_cb=iter_cb)


# In[11]:


xs = np.array(intermediate_values)[:, 0]
ys = np.array(intermediate_values)[:, 1]
zs = np.array(intermediate_values)[:, 2]
print(xs[len(xs)-1])


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, x)
plt.figure(figsize=(8, 6), dpi=80)
ax = plt.subplot(111, projection='3d')
ax.plot_surface(X, Y, rosen([X, Y]))
ax.plot(xs, ys, zs, 'ro')
ax.set_xlim([-2, 4])
ax.set_ylim([-3, 4])
ax.set_zlim([0, 1600])
ax.view_init()

ax.view_init(10, -90)

plt.show()


# In[ ]:




