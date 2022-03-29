
from functools import cache
import numpy as np
import numba as nb


@nb.njit(parallel=True)
def update_IRF_hh(IRF_pols,dpols,IRF):

    T = dpols.shape[0]
    Nfix = dpols.shape[1]
    Nz = dpols.shape[2]
    Nendodim = dpols.ndim-3
    
    if Nendodim == 1:
        
        Nendo1 = dpols.shape[3]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_endo1 in nb.prange(Nendo1):
                    for t in range(T):
                        IRF_pols[i_fix,i_z,i_endo1,t] += np.sum(dpols[:T-t,0,i_z,i_endo1]*IRF[t:])

    else:

        raise ValueError('too many dimensions')

@nb.njit(parallel=True)
def simulate_agg(epsilons,IRF_mat,sim_mat):

    Nvars = IRF_mat.shape[0]
    Nshocks = IRF_mat.shape[1]
    T = IRF_mat.shape[2]

    simT = epsilons.shape[1] 
    
    for i_var in nb.prange(Nvars):
        for t in range(simT):
            sim_mat[i_var,t] = 0.0
            for i_shock in range(Nshocks):
                for s in range(T):
                    if t-s >= 0:
                        sim_mat[i_var,t] += IRF_mat[i_var,i_shock,s]*epsilons[i_shock,t-s]

@nb.njit(parallel=True,cache=False)
def simulate_agg_hh(epsilons,IRF_pols_mat,sim_pols_mat):

    Npols = IRF_pols_mat.shape[0]
    Nshocks = IRF_pols_mat.shape[1]
    Nfix = IRF_pols_mat.shape[2]
    Nz = IRF_pols_mat.shape[3]
    T = IRF_pols_mat.shape[-1]
    Ndim = IRF_pols_mat.ndim-5

    simT = epsilons.shape[1]

    if Ndim == 1:
        
        Nendo1 = IRF_pols_mat.shape[4]

        for i_pol in nb.prange(Npols):
            for i_fix in nb.prange(Nfix):
                for i_z in nb.prange(Nz):            
                    for i_endo1 in range(Nendo1):                    
                        for t in range(simT):
                            for i_shock in range(Nshocks):
                                for s in range(T):     
                                    if t-s >= 0:
                                        IRF_ = IRF_pols_mat[i_pol,i_shock,i_fix,i_z,i_endo1,s]
                                        sim_pols_mat[i_pol,t,i_fix,i_z,i_endo1] += IRF_*epsilons[i_shock,t-s]

    else:

        raise ValueError('too many dimensions')