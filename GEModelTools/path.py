# contains functions for working with the transition path

import numpy as np
import numba as nb

@nb.njit
def lag(first,after):
    return np.hstack((np.array([first]),after))

@nb.njit
def lead(before,last):
    return np.hstack((before,np.array([last])))