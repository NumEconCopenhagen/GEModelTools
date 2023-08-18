# contains functions for working with the transition path

import importlib
import numpy as np
import numba as nb

def get_varnames(blockstr):

    modulename,funcname = blockstr.split('.')
    module = importlib.import_module(modulename)
    func = eval(f'module.{funcname}')

    n = func.__code__.co_argcount
    varnames = func.__code__.co_varnames[3:n] 
    
    return varnames

@nb.njit
def lag(inivalue,pathvalue):

    output = np.empty_like(pathvalue)
    output[0,:] = inivalue
    output[1:,:] = pathvalue[:-1,:]
    return output

@nb.njit
def lead(pathvalue,ssvalue):

    output = np.empty_like(pathvalue)
    output[:-1,:] = pathvalue[1:,:]
    output[-1,:] = ssvalue
    return output

@nb.njit
def prev(x,t,inivalue):
    if t > 0:
        return x[t-1]
    else:
        return np.repeat(inivalue,x.shape[1])
    
@nb.njit
def next(x,t,ssvalue):
    if t+1 < x.shape[0]:
        return x[t+1]
    else:
        return np.repeat(ssvalue,x.shape[1])

@nb.njit
def isclose(x,y):
    return np.abs(x-y) < 1e-8 
    
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