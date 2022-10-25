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

                # d. avoid extrapolation
                w[i_fix,i_z,i_endo] = np.fmin(w[i_fix,i_z,i_endo],1.0)
                w[i_fix,i_z,i_endo] = np.fmax(w[i_fix,i_z,i_endo],0.0)

@nb.njit
def find_i_and_w_1d_1d_path(T,path_pol1,grid1,path_i,path_w):
    """ find indices and weights for simulation along transition path"""

    for k in range(T):

        t = (T-1)-k

        find_i_and_w_1d_1d(path_pol1[t],grid1,path_i[t],path_w[t])

    return k

@nb.njit(parallel=True)   
def simulate_hh_forwards_endo(D,i,w,Dbeg_plus):
    """ simulate endougenous deterministic transition given indices and weights """
    
    Nfix = D.shape[0]
    Nz = D.shape[1]
    Ndim = D.ndim-2

    if Ndim == 1:
        
        Nendo1 = D.shape[2]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
            
                Dbeg_plus[i_fix,i_z,:] = 0.0
                for i_endo in range(Nendo1):
                    
                    # i. from
                    D_ = D[i_fix,i_z,i_endo]

                    # ii. to
                    i_ = i[i_fix,i_z,i_endo]            
                    w_ = w[i_fix,i_z,i_endo]
                    Dbeg_plus[i_fix,i_z,i_] += D_*w_
                    Dbeg_plus[i_fix,i_z,i_+1] += D_*(1.0-w_)
    
    else:

        raise ValueError('too many dimensions')

@nb.njit(parallel=True)   
def simulate_hh_forwards_exo(Dbeg,z_trans_T,D):
    """ exogenous stochastic transition given transition matrix """
    
    Nfix = Dbeg.shape[0]

    for i_fix in nb.prange(Nfix):
        D[i_fix] = z_trans_T[i_fix]@Dbeg[i_fix]

@nb.njit(parallel=True)   
def simulate_hh_forwards_exo_transpose(Dbeg,z_trans,D):
    """ simulate given indices and weights """

    Nfix = Dbeg.shape[0]

    for i_fix in nb.prange(Nfix):
        D[i_fix] = z_trans[i_fix]@Dbeg[i_fix]
    
@nb.njit(parallel=True)   
def simulate_hh_forwards_endo_transpose(Dbeg_plus,i,w,D):
    """ simulate given indices and weights """

    Nfix = Dbeg_plus.shape[0]
    Nz = Dbeg_plus.shape[1]
    Ndim = Dbeg_plus.ndim-2    

    if Ndim == 1:
        
        Nendo1 = D.shape[2]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_endo in nb.prange(Nendo1):
                    i_ = i[i_fix,i_z,i_endo]
                    w_ = w[i_fix,i_z,i_endo]
                    D[i_fix,i_z,i_endo] = w_*Dbeg_plus[i_fix,i_z,i_] + (1.0-w_)*Dbeg_plus[i_fix,i_z,i_+1]

    else:

        raise ValueError('too many dimensions')

@nb.njit
def simulate_hh_ss(par,ss):
    """ simulate forwards to steady state """

    it = 0
    z_trans_T = np.transpose(ss.z_trans,axes=(0,2,1)).copy()

    while True:
        
        old_D = ss.D.copy()

        # i. exogenous update
        simulate_hh_forwards_exo(ss.Dbeg,z_trans_T,ss.D)

        # ii. check convergence
        if it > 1 and np.max(np.abs(ss.D-old_D)) < par.tol_simulate: 
            return it

        # important for fake news algorithm: 
        # Dbeg and D is related by an exogenous forward update 

        # iii. endogenous update
        simulate_hh_forwards_endo(ss.D,ss.pol_indices,ss.pol_weights,ss.Dbeg)

        # iii. increment
        it += 1
        if it > par.max_iter_simulate: 
            raise ValueError('simulate_hh_ss(), too many iterations')

@nb.njit
def simulate_hh_path(par,path):
    """ simulate along transition path """

    for t in range(par.T):

        Dbeg = path.Dbeg[t]
        D = path.D[t]
        z_trans_T = np.transpose(path.z_trans[t],axes=(0,2,1)).copy()

        # a. exogenous update
        simulate_hh_forwards_exo(Dbeg,z_trans_T,D)

        # b. endogenous update
        if t < par.T-1:

            Dbeg_plus = path.Dbeg[t+1]
            i = path.pol_indices[t]
            w = path.pol_weights[t]

            simulate_hh_forwards_endo(D,i,w,Dbeg_plus)

@nb.njit
def simulate_hh_z_path(par,z_trans,Dz,Dz_ini):
    """ find steady state for exogenous distribution """

    for t in range(par.T):

        # a. lagged
        if t == 0:
            Dz_lag = Dz_ini
        else:                
            Dz_lag = Dz[t-1]

        # b. transpose
        z_trans_T = np.transpose(z_trans[t],axes=(0,2,1)).copy()

        # c. update
        for i_fix in range(par.Nfix):
            Dz[t,i_fix] = z_trans_T[i_fix]@Dz_lag[i_fix]

@nb.njit
def simulate_hh_forwards(D,i,w,z_trans_T,D_plus):

    Dbeg_plus = np.zeros(D.shape)
    simulate_hh_forwards_endo(D,i,w,Dbeg_plus)
    simulate_hh_forwards_exo(Dbeg_plus,z_trans_T,D_plus)