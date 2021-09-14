# contains simulation functions
 
import numpy as np
import numba as nb
from consav.linear_interp import binary_search

@nb.njit(parallel=True) 
def find_i_and_w_1d_1d(pol1,grid1,i,w):
    """ find indices and weights for simulation """

    Nfix = pol1.shape[0]
    Nz = pol1.shape[1]
    Nendo1 = pol1.shape[2]

    for i_z in nb.prange(Nz):
        for i_fix in nb.prange(Nfix):
            for i_endo in nb.prange(Nendo1):
                
                # a. policy
                pol1_ = pol1[i_fix,i_z,i_endo]

                # b. find i_ such a_grid[i_] <= a_ < a_grid[i_+1]
                i_ = i[i_fix,i_z,i_endo] = binary_search(0,grid1.size,grid1,pol1_) 

                # c. weight
                w[i_fix,i_z,i_endo] = (grid1[i_+1] - pol1_) / (grid1[i_+1] - grid1[i_])

                # d. bound simulation at upper grid point
                w[i_fix,i_z,i_endo] = np.fmin(w[i_fix,i_z,i_endo],1.0)

@nb.njit(parallel=True)   
def simulate_hh_forwards(D_lag,i,w,z_trans_T,D):
    """ simulate given indices and weights """

    # a. assuming z is constant 
    # (same as multiplication with Q transposed)
    
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Ndim = D.ndim-2

    if Ndim == 1:
        
        Nendo1 = D.shape[2]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
            
                D[i_fix,i_z,:] = 0
                for i_endo in range(Nendo1):
                    
                    # i. from
                    D_ = D_lag[i_fix,i_z,i_endo]

                    # i. to
                    i_ = i[i_fix,i_z,i_endo]            
                    w_ = w[i_fix,i_z,i_endo]
                    D[i_fix,i_z,i_] += D_*w_
                    D[i_fix,i_z,i_+1] += D_*(1.0-w_)
    
    
    else:

        raise ValueError('too many dimensions')

    # b. account for transition of z
    # (same as multiplication with tilde Pi transposed)
    for i_fix in nb.prange(Nfix):
        D[i_fix] = z_trans_T@D[i_fix].copy()

@nb.njit(parallel=True)   
def simulate_hh_forwards_transpose(D_lag,i,w,z_trans,D):
    """ simulate given indices and weights """

    Nfix = D.shape[0]
    Nz = D.shape[1]
    Ndim = D.ndim-2

    # a. account for transition z
    # (same as multiplication with tilde Pi)
    D_temp = np.zeros(D.shape)
    for i_fix in nb.prange(Nfix):
        D_temp[i_fix] = z_trans@D_lag[i_fix].copy()
    
    # b. assuming z is constant
    # (same as multiplication with Q)    

    if Ndim == 1:
        
        Nendo1 = D.shape[2]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_endo in nb.prange(Nendo1):
                    i_ = i[i_fix,i_z,i_endo]
                    w_ = w[i_fix,i_z,i_endo]
                    D[i_fix,i_z,i_endo] = w_*D_temp[i_fix,i_z,i_] + (1.0-w_)*D_temp[i_fix,i_z,i_+1]

    else:

        raise ValueError('too many dimensions')

@nb.njit
def simulate_hh_ss(par,sol,sim):
    """ simulate forward to steady state """

    # a. prepare
    z_trans_T = par.z_trans_ss.T.copy()
    D_lag = np.zeros(sim.D.shape)

    # b. iterate
    it = 0
    while True:
        
        # i. update distribution
        D_lag = sim.D.copy()
        simulate_hh_forwards(D_lag,sol.i,sol.w,z_trans_T,sim.D)

        # ii. check convergence
        if np.max(np.abs(sim.D-D_lag)) < par.tol_simulate: 
            return it

        # iii. increment
        it += 1
        if it > par.max_iter_simulate: 
            raise ValueError('simulate_hh_ss(), too many iterations')

@nb.njit
def simulate_hh_path(par,sol,sim):
    """ simulate along path """

    for t in range(par.transition_T):
        
        z_trans_T = par.z_trans_path[t].T.copy()

        # a. lag
        if t == 0:
            D_lag = sim.D # steadt state
        else:
            D_lag = sim.path_D[t-1]

        # b. unpack
        path_i = sol.path_i[t]
        path_w = sol.path_w[t]

        # c. update distribution
        simulate_hh_forwards(D_lag,path_i,path_w,z_trans_T,sim.path_D[t])

