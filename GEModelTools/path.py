# contains functions for working with the transition path

import numpy as np
import numba as nb

@nb.njit
def lag(ssvalue,pathvalue):
    return np.hstack((np.array([ssvalue]),pathvalue[:-1]))

@nb.njit
def lead(pathvalue,ssvalue):
    return np.hstack((pathvalue[1:],np.array([ssvalue])))

@nb.njit
def bound(var,a,b):
    return np.fmin(np.fmax(var,a),b)

@nb.njit
def bisection(f,a,b,args=(),max_iter=500,tol=1e-12):
      
    i = 0
    while i < max_iter:
        
        m = (a+b)/2
        fm = f(m,*args)
        
        if abs(fm) < tol:
            break        
        elif f(a,*args)*fm < 0:
            b = m
        elif f(b,*args)*fm < 0:
            a = m
       
        i += 1
        
    return m    