import numpy as np
import numba as nb

from consav.linear_interp import binary_search

########
# find #
########

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
                w[i_fix,i_z,i_endo] = (grid1[i_+1]-pol1_)/(grid1[i_+1]-grid1[i_])

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
def find_i_and_w_2d_1d(pol1,grid_endo,grid1,grid2,i,w):
    """ find indices and weights for simulation """

    Nfix = pol1.shape[0]
    Nz = pol1.shape[1]
    Nendo1 = grid1.size
    Nendo2 = grid2.size

    for i_z in nb.prange(Nz):
        for i_fix in nb.prange(Nfix):
            for i_endo1 in nb.prange(Nendo1):
                for i_endo2 in nb.prange(Nendo2):

                    # a. policy
                    pol1_ = pol1[i_fix,i_z,i_endo1,i_endo2]

                    # b. find i_ such l_grid[i_] <= l_ < l_grid[i_+1] given l_grid[i_endo]
                    i_ = i[i_fix,i_z,i_endo1,i_endo2] = binary_search(0,grid_endo.size,grid_endo,pol1_)

                    # c. weight
                    w[i_fix,i_z,i_endo1,i_endo2] = (grid_endo[i_+1]-pol1_)/(grid_endo[i_+1]-grid_endo[i_])

                    # d. avoid extrapolation
                    w[i_fix,i_z,i_endo1,i_endo2] = np.fmin(w[i_fix,i_z,i_endo1,i_endo2],1.0)
                    w[i_fix,i_z,i_endo1,i_endo2] = np.fmax(w[i_fix,i_z,i_endo1,i_endo2],0.0)

@nb.njit
def find_i_and_w_2d_1d_path(T,path_pol1,grid_endo,grid1,grid2,path_i,path_w):
    """ find indices and weights for simulation along transition path"""

    for k in range(T):

        t = (T-1)-k
        find_i_and_w_2d_1d(path_pol1[t],grid_endo,grid1,grid2,path_i[t],path_w[t])

    return k


#######
# exo #
#######

@nb.njit(parallel=True)
def simulate_hh_forwards_exo(Dbeg,z_trans_T,D):
    """ exogenous stochastic transition given transition matrix """

    Nfix = Dbeg.shape[0]

    if Dbeg.ndim < 4:
    
        Nz = Dbeg.shape[1]
        Nendo1 = Dbeg.shape[2]
            
        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_endo1 in nb.prange(Nendo1):

                    D[i_fix,i_z,i_endo1] = 0.0
                    for i_z_lag in range(Nz):
                        D[i_fix,i_z,i_endo1] += z_trans_T[i_fix,i_z,i_z_lag]*Dbeg[i_fix,i_z_lag,i_endo1]

    elif Dbeg.ndim == 4:

        Nz = Dbeg.shape[1]
        Nendo1 = Dbeg.shape[2]
        Nendo2 = Dbeg.shape[3]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_endo1 in nb.prange(Nendo1):
                    for i_endo2 in nb.prange(Nendo2):

                        D[i_fix,i_z,i_endo1,i_endo2] = 0.0
                        for i_z_lag in range(Nz):
                            D[i_fix,i_z,i_endo1,i_endo2] += z_trans_T[i_fix,i_z,i_z_lag]*Dbeg[i_fix,i_z_lag,i_endo1,i_endo2]

    else:

        raise NotImplementedError

@nb.njit(parallel=True)
def simulate_hh_forwards_exo_transpose(Dbeg,z_trans):
    """ simulate given indices and weights """

    Nfix = Dbeg.shape[0]
    D = np.zeros_like(Dbeg)

    if Dbeg.ndim < 4:

        Nz = Dbeg.shape[1]
        Nendo1 = Dbeg.shape[2]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_endo1 in nb.prange(Nendo1):

                    for i_z_lag in range(Nz):
                        D[i_fix,i_z,i_endo1] += z_trans[i_fix,i_z,i_z_lag]*Dbeg[i_fix,i_z_lag,i_endo1]

    elif Dbeg.ndim == 4:

        Nz = Dbeg.shape[1]
        Nendo1 = Dbeg.shape[2]
        Nendo2 = Dbeg.shape[3]

        for i_fix in nb.prange(Nfix):
            for i_z in nb.prange(Nz):
                for i_endo1 in nb.prange(Nendo1):
                    for i_endo2 in nb.prange(Nendo2):

                        for i_z_lag in range(Nz):
                            D[i_fix,i_z,i_endo1,i_endo2] += z_trans[i_fix,i_z,i_z_lag]*Dbeg[i_fix,i_z_lag,i_endo1,i_endo2]
    else:

        raise NotImplementedError

    return D


########
# endo #
########

@nb.njit(parallel=True)
def simulate_hh_forwards_endo_1d(D,i,w,Dbeg_plus):
    """ forward simulation with 1d distribution """

    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):

            Dbeg_plus[i_fix,i_z,:] = 0.0

            for i_endo in range(Nendo1):

                # i. from
                D_ = D[i_fix,i_z,i_endo]

                # ii. to
                i_ = i[0,i_fix,i_z,i_endo]
                w_ = w[0,i_fix,i_z,i_endo]

                Dbeg_plus[i_fix,i_z,i_] += D_*w_
                Dbeg_plus[i_fix,i_z,i_+1] += D_*(1.0-w_)

@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d(D,i,w,Dbeg_plus):
    """ forward simulation with 2d distribution along both grid dimension """

    Nfix = D.shape[0]
    Nz = D.shape[1]
    Nendo1 = D.shape[2]
    Nendo2 = D.shape[3]

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):

            Dbeg_plus[i_fix,i_z,:,:] = 0.0

            for i_endo1 in range(Nendo1):
                for i_endo2 in range(Nendo2):

                    # i. from
                    D_ = D[i_fix,i_z,i_endo1,i_endo2]

                    # ii. to
                    i_1_ = i[0,i_fix,i_z,i_endo1,i_endo2]
                    i_2_ = i[1,i_fix,i_z,i_endo1,i_endo2]
                    w_1_ = w[0,i_fix,i_z,i_endo1,i_endo2]
                    w_2_ = w[1,i_fix,i_z,i_endo1,i_endo2]

                    Dbeg_plus[i_fix,i_z,i_1_,i_2_] += w_1_*w_2_*D_
                    Dbeg_plus[i_fix,i_z,i_1_+1,i_2_] += (1-w_1_)*w_2_*D_
                    Dbeg_plus[i_fix,i_z,i_1_,i_2_+1] += w_1_*(1-w_2_)*D_
                    Dbeg_plus[i_fix,i_z,i_1_+1,i_2_+1] += (1-w_1_)*(1-w_2_)*D_

@nb.njit
def simulate_hh_forwards_endo(D,i,w,Dbeg_plus):
    """ replaced function to simulate endougenous deterministic transition given indices and weights """

    Ndim = D.ndim

    if Ndim == 3:
        simulate_hh_forwards_endo_1d(D,i,w,Dbeg_plus)
    elif Ndim == 4:
        simulate_hh_forwards_endo_2d(D,i,w,Dbeg_plus)
    else:
        raise NotImplementedError


################
# endo - trans #
################

@nb.njit(parallel=True)
def simulate_hh_forwards_endo_1d_trans(Dbeg_plus,i,w):
    """ forward simulation with 1d distribution """

    Nfix = Dbeg_plus.shape[0]
    Nz = Dbeg_plus.shape[1]
    Nendo1 = Dbeg_plus.shape[2]

    D = np.zeros_like(Dbeg_plus)

    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            for i_endo in nb.prange(Nendo1):

                i_ = i[0,i_fix,i_z,i_endo]
                w_ = w[0,i_fix,i_z,i_endo]

                L = w_*Dbeg_plus[i_fix,i_z,i_]
                R = (1.0-w_)*Dbeg_plus[i_fix,i_z,i_+1]

                D[i_fix,i_z,i_endo] = L+R

    return D

@nb.njit(parallel=True)
def simulate_hh_forwards_endo_2d_trans(Dbeg_plus,i,w):
    """ forward simulation with 2d distribution along both grid dimension """

    Nfix = Dbeg_plus.shape[0]
    Nz = Dbeg_plus.shape[1]
    Nendo1 = Dbeg_plus.shape[2]
    Nendo2 = Dbeg_plus.shape[3]

    D = np.zeros_like(Dbeg_plus)
    for i_fix in nb.prange(Nfix):
        for i_z in nb.prange(Nz):
            for i_endo1 in nb.prange(Nendo1):
                for i_endo2 in nb.prange(Nendo2):

                    i_1_ = i[0,i_fix,i_z,i_endo1,i_endo2]
                    i_2_ = i[1,i_fix,i_z,i_endo1,i_endo2]
                    w_1_ = w[0,i_fix,i_z,i_endo1,i_endo2]
                    w_2_ = w[1,i_fix,i_z,i_endo1,i_endo2]

                    LL =  w_1_*w_2_*Dbeg_plus[i_fix,i_z,i_1_,i_2_]
                    LR = w_1_*(1-w_2_)*Dbeg_plus[i_fix,i_z,i_1_,i_2_+1]
                    RL = (1-w_1_)*w_2_*Dbeg_plus[i_fix,i_z,i_1_+1,i_2_]
                    RR = (1-w_1_)*(1-w_2_)*Dbeg_plus[i_fix,i_z,i_1_+1,i_2_+1]

                    D[i_fix,i_z,i_endo1,i_endo2] = LL + LR + RL + RR
    return D

@nb.njit
def simulate_hh_forwards_endo_transpose(Dbeg_plus,i,w):
    """ simulate given indices and weights """

    Ndim_i = i.ndim
    Ndim_D = Dbeg_plus.ndim

    if Ndim_D == 3:
        D = simulate_hh_forwards_endo_1d_trans(Dbeg_plus,i,w)
    elif Ndim_D == 4 and Ndim_D+1 == Ndim_i:
        D = simulate_hh_forwards_endo_2d_trans(Dbeg_plus,i,w)
    else:
        raise NotImplementedError
    return D


############
# combined #
############

@nb.njit
def simulate_hh_forwards(D,i,w,z_trans_T,D_plus):

    Dbeg_plus = np.zeros(D.shape)
    simulate_hh_forwards_endo(D,i,w,Dbeg_plus)
    simulate_hh_forwards_exo(Dbeg_plus,z_trans_T,D_plus)

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
            raise ValueError('simulate_hh_ss(),too many iterations')

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
