# contains main GEModelClass

from re import A
import time
from copy import deepcopy
import numpy as np

from EconModel import jit
from consav.misc import elapsed

from . import tests
from .simulate_hh import find_i_and_w_1d_1d, find_i_and_w_1d_1d_path
from .simulate_hh import simulate_hh_D0, simulate_hh_forwards, simulate_hh_forwards_transpose
from .simulate_hh import simulate_hh_path, simulate_hh_ss
from .broyden_solver import broyden_solver
from .simulate import update_IRF_hh,simulate_agg,simulate_agg_hh
from .figures import show_IRFs

class GEModelClass:

    ############
    # 1. setup #
    ############

    def allocate_GE(self,sol_shape,update_hh=True,ss_nan=True):
        """ allocate GE variables """

        par = self.par
        sol = self.sol
        sim = self.sim

        #Nfix = sol_shape[0]
        Nz = sol_shape[1]
        Nendos = sol_shape[2:]

        path_sol_shape = (par.T,*sol_shape)
        
        # a. defaults
        par.__dict__.setdefault('T',500)
        par.__dict__.setdefault('max_iter_solve',50_000)
        par.__dict__.setdefault('max_iter_simulate',50_000)
        par.__dict__.setdefault('max_iter_broyden',100)
        par.__dict__.setdefault('tol_solve',1e-12)
        par.__dict__.setdefault('tol_simulate',1e-12)
        par.__dict__.setdefault('tol_broyden',1e-10)

        for varname in self.shocks:
            par.__dict__.setdefault(f'jump_{varname}',0.0)
            par.__dict__.setdefault(f'rho_{varname}',0.0)

        assert hasattr(self.par,'Nz'), 'par.Nz must be specified'

        # b. checks
        assert Nz == par.Nz, f'sol_shape is wrong, sol_shape[1] = {Nz}, par.Nz = {par.Nz}'

        attrs = ['grids_hh','pols_hh','inputs_hh','intertemps_hh']
        attrs += ['shocks','unknowns','targets','varlist']
        attrs += ['par','sol','sim','ss','path']
        for attr in attrs:
            assert hasattr(self,attr), f'missing .{attr}'

        for attr in attrs + ['jac','H_U','H_Z','jac_hh','IRF']:
            if not attr in self.other_attrs:
                self.other_attrs.append(attr)

        for i,(varname,Nendo) in enumerate(zip(self.grids_hh,Nendos)):
            Nx = f'N{varname}'
            assert hasattr(par,Nx), f'{Nx} not in .par'
            Nxval = getattr(par,Nx)
            assert Nendo == Nxval, f'sol_shape is wrong, sol_shape[{i}] = {Nendo}, par.{Nx} = {Nxval}'
                
        for varname in self.inputs_hh:
            assert varname in self.varlist, f'{varname} not in .varlist'

        for varname in self.outputs_hh:
            varname_agg = f'{varname.upper()}_hh'
            assert varname_agg in self.varlist, f'{varname_agg} not in .varlist'
            hasattr(self.path,varname_agg), f'{varname_agg} not in .path'

        for varname in self.shocks + self.unknowns + self.targets:
            assert varname in self.varlist, f'{varname} not in .varlist'

        # c. allocate grids and transition matrices
        if update_hh:

            for varname in self.grids_hh:

                gridname = f'{varname}_grid' 
                Nx = getattr(par,f'N{varname}')
                gridarray = np.zeros(Nx)
                setattr(par,gridname,gridarray)
                
            par.z_grid_ss = np.zeros(par.Nz)
            par.z_trans_ss = np.zeros((par.Nz,par.Nz))
            par.z_ergodic_ss = np.zeros(par.Nz)
            
            par.z_grid_path = np.zeros((par.T,par.Nz))
            par.z_trans_path = np.zeros((par.T,par.Nz,par.Nz))        

        # d. allocate household variables
        if update_hh:

            for varname in self.outputs_hh + self.intertemps_hh:

                assert not varname == 'i', f'sol.{varname} not allowed'
                assert not varname == 'w', f'sol.{varname} not allowed'

                array = np.zeros(sol_shape)
                setattr(sol,varname,array)
                array_path = np.zeros(path_sol_shape)
                setattr(sol,f'path_{varname}',array_path)

            sol.i = np.zeros(sol_shape,dtype=np.int_)
            sol.path_i = np.zeros(path_sol_shape,dtype=np.int_)
            sol.w = np.zeros(sol_shape)
            sol.path_w = np.zeros(path_sol_shape)

            # e. allocate distribution
            sim.D = np.zeros(sol_shape)
            sim.path_D = np.zeros(path_sol_shape)

        # f. allocate path variables
        path_shape = (len(self.unknowns)*par.T,par.T)
        for varname in self.varlist:
            if ss_nan: setattr(self.ss,varname,np.nan)
            setattr(self.path,varname,np.zeros(path_shape))

        # g. allocate Jacobians
        if update_hh:
            self.jac_hh = {}

        self.jac = {}
        for outputname in self.varlist:
            for inputname in self.unknowns+self.shocks:
                key = (outputname,inputname)
                self.jac[key] = np.zeros((par.T,par.T))

        self.H_U = np.zeros((
            len(self.targets)*par.T,
            len(self.unknowns)*par.T))

        self.H_Z = np.zeros((
            len(self.targets)*par.T,
            len(self.shocks)*par.T))

        self.G_U = np.zeros(self.H_Z.shape)

        self.IRF = {}
        for varname in self.varlist:
            self.IRF[varname] = np.repeat(np.nan,par.T)
            for shockname in self.shocks:
                self.IRF[(varname,shockname)] = np.repeat(np.nan,par.T)

        # i. allocate path variables
        if hasattr(par,'simT'):
            for varname in self.varlist:
                
                assert not varname == 'D', f'sim.{varname} not allowed'
                assert not varname == 'path_D', f'sim.{varname} not allowed'

                setattr(self.sim,f'd{varname}',np.zeros(par.simT))

    def create_grids(self):
        """ create grids """

        raise NotImplementedError

    def create_grids_path(self):
        """ create grids for transition path (z_grid_path and z_trans_path) """

        raise NotImplementedError

    def print_unpack_varlist(self):
        """ print varlist for use in evaluate_path() """

        print(f'    for thread in nb.prange(threads):\n')
        print('        # unpack')
        for varname in self.varlist:
            print(f'        {varname} = path.{varname}[thread,:]')

    def update_aggregate_settings(self,shocks=None,unknowns=None,targets=None):
        """ update aggregate settings and re-allocate jac etc. """
        
        if not shocks is None: self.shocks = shocks
        if not unknowns is None: self.unknowns = unknowns
        if not targets is None: self.targets = targets

        sol_shape = self.sol.i.shape
        self.allocate_GE(sol_shape,update_hh=False,ss_nan=False)

    ####################
    # 2. steady state #
    ###################

    def _find_i_and_w(self):
        """ find indices and weigths for simulation """

        par = self.par
        sol = self.sol

        if len(self.grids_hh) == 1:
            pol1 = getattr(sol,f'{self.grids_hh[0]}')
            grid1 = getattr(par,f'{self.grids_hh[0]}_grid') 
            find_i_and_w_1d_1d(pol1,grid1,sol.i,sol.w)
        else:
            raise NotImplementedError

    def solve_hh_ss(self,do_print=False):
        """ solve the household problem in steady state """

        t0 = time.time()

        # a. create grids
        self.create_grids()

        # b. solve backwards until convergence
        with jit(self) as model:
            
            par = model.par
            sol = model.sol
            ss = model.ss

            it = 0
            while True:

                # i. old policy
                old = {pol:getattr(sol,pol).copy() for pol in self.pols_hh}

                # ii. step backwards
                step_vars = {'par':par,'z_grid':par.z_grid_ss,'z_trans_plus':par.z_trans_ss}
                for varname in self.inputs_hh: step_vars[varname] = getattr(ss,varname)
                for varname in self.outputs_hh + self.intertemps_hh: step_vars[varname] = getattr(sol,varname)
                for varname in self.intertemps_hh: step_vars[f'{varname}_plus'] = getattr(sol,varname)
        
                self.solve_hh_backwards(**step_vars)

                # iii. check change in policy
                max_abs_diff = max([np.max(np.abs(getattr(sol,pol)-old[pol])) for pol in self.pols_hh])
                if max_abs_diff < par.tol_solve: 
                    break
                
                # iv. increment
                it += 1
                if it > par.max_iter_solve: 
                    raise ValueError('solve_hh_ss(), too many iterations')

        if do_print: print(f'household problem in ss solved in {elapsed(t0)} [{it} iterations]')

        # c. indices and weights    
        self._find_i_and_w()

    def simulate_hh_ss(self,do_print=False,find_i_and_w=False):
        """ simulate the household problem in steady state """
        
        par = self.par
        sol = self.sol
        sim = self.sim  

        t0 = time.time()

        if find_i_and_w: self._find_i_and_w()

        # a. initial guess
        basepol = getattr(sol,self.pols_hh[0])
        sim.D = np.zeros(basepol.shape)

        Nfix = basepol.shape[0]
        for i_fix in range(Nfix):
            if len(self.grids_hh) == 1:
                sim.D[i_fix,:,0] = par.z_ergodic_ss/Nfix
            elif len(self.grids_hh) == 2:
                sim.D[i_fix,:,0,0] = par.z_ergodic_ss/Nfix
            elif len(self.grids_hh) == 3:
                sim.D[i_fix,:,0,0,0] = par.z_ergodic_ss/Nfix
            else:
                raise NotImplementedError
    
        # b. simulate
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim
            
            it = simulate_hh_ss(par,sol,sim)

        if do_print: print(f'household problem in ss simulated in {elapsed(t0)} [{it} iterations]')

    def find_ss(self,do_print=False):
        """ solve for the steady state """

        raise NotImplementedError

    #####################
    # 3. household path #
    #####################

    def _find_i_and_w_path(self):
        """ find indices and weights for simulation along the transition path"""

        par = self.par
        sol = self.sol

        if len(self.grids_hh) == 1:
            path_pol1 = getattr(sol,f'path_{self.grids_hh[0]}')
            grid1 = getattr(par,f'{self.grids_hh[0]}_grid') 
            find_i_and_w_1d_1d_path(par.T,path_pol1,grid1,sol.path_i,sol.path_w)
        else:
            raise NotImplemented

    def solve_hh_path(self,do_print=False):
        """ gateway for solving the household problem along the transition path """

        t0 = time.time()

        # a. create grids
        self.create_grids_path()

        # b. solve backwards
        with jit(self) as model:

            par = model.par
            sol = model.sol
            path = model.path
            
            for k in range(par.T):

                t = (par.T-1)-k

                # i. variables
                step_vars = {'par':par,'z_grid':par.z_grid_path[t]}
                for varname in self.inputs_hh: step_vars[varname] = getattr(path,varname)[0,t]
                for varname in self.outputs_hh + self.intertemps_hh: step_vars[varname] = getattr(sol,f'path_{varname}')[t]
                
                if t == par.T-1:
                    step_vars['z_trans_plus'] = par.z_trans_ss
                    for varname in self.intertemps_hh: step_vars[f'{varname}_plus'] = getattr(sol,varname)
                else:
                    step_vars['z_trans_plus'] = par.z_trans_path[t+1]
                    for varname in self.intertemps_hh: step_vars[f'{varname}_plus'] = getattr(sol,f'path_{varname}')[t+1]

                # ii. step backwards
                self.solve_hh_backwards(**step_vars)

        # c. indices and weights
        self._find_i_and_w_path()

        if do_print: print(f'household problem solved along transition path in {elapsed(t0)}')

    def simulate_hh_path(self,do_print=False,find_i_and_w=False):
        """ gateway for simulating the household problem along the transition path"""
        
        t0 = time.time() 

        if find_i_and_w: self._find_i_and_w_path()

        with jit(self) as model:
            simulate_hh_path(model.par,model.sol,model.sim)

        if do_print: print(f'household problem simulated along transition in {elapsed(t0)}')

    def _set_inputs_hh_ss(self):
        """ set household inputs to steady state """

        for inputname in self.inputs_hh:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    ################
    # 4. Jacobians #
    ################

    def _set_shocks_ss(self):
        """ set shocks to steady state """

        for inputname in self.shocks:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    def _set_unknowns_ss(self):
        """ set unknwnos to steady state """

        for inputname in self.unknowns:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    def _set_unknowns(self,x,inputs,parallel=False):
        """ set unknowns """

        par = self.par
        path = self.path

        if parallel:

            Ninputs = len(inputs)
            Ninputs_tot = Ninputs*self.par.T

            x = x.reshape((len(inputs),par.T,Ninputs_tot))
            for i,varname in enumerate(inputs):
                array = getattr(path,varname)                    
                array[:Ninputs_tot,:] = x[i,:,:].T

        else:

            if not x is None:
                x = x.reshape((len(inputs),par.T))
                for i,varname in enumerate(inputs):
                    array = getattr(path,varname)                    
                    array[0,:] = x[i,:]

    def _get_errors(self,inputs=None,parallel=False):
        """ get errors from targets """
        
        if parallel:

            assert not inputs is None

            Ninputs = len(inputs)
            Ninputs_tot = Ninputs*self.par.T

            errors = np.zeros((len(self.targets),self.par.T,Ninputs_tot))
            for i,varname in enumerate(self.targets):
                errors[i,:,:] = getattr(self.path,varname)[:Ninputs_tot,:].T

        else:

            errors = np.zeros((len(self.targets),self.par.T))
            for i,varname in enumerate(self.targets):
                errors[i,:] = getattr(self.path,varname)[0,:]

        return errors

    def _calc_jac_hh_direct(self,jac_hh,inputname,dx=1e-4,do_print=False,s_list=None):
        """ compute Jacobian of household problem """

        par = self.par
        sol = self.sol
        sim = self.sim
        
        if s_list is None: s_list = list(range(par.T))

        t0 = time.time()
        if do_print: print(f'finding Jacobian wrt. {inputname:3s}:',end='')
            
        # a. allocate
        for outputname in self.outputs_hh:
            jac_hh[(f'{outputname.upper()}_hh',inputname)] = np.zeros((par.T,par.T))

        # b. solve with shock in last period
        self._set_inputs_hh_ss()

        if not inputname == 'ghost':
            shockarray = getattr(self.path,inputname)
            shockarray[0,-1] += dx

        self.solve_hh_path()

        # c. simulate
        par_shock = deepcopy(self.par)
        sol_shock = deepcopy(self.sol)

        for s in s_list:

            if do_print: print(f' {s}',end='')
            
            # i. before shock only time to shock matters
            par.z_grid_path[:s+1] = par_shock.z_grid_path[par.T-(s+1):]
            par.z_trans_path[:s+1] = par_shock.z_trans_path[par.T-(s+1):]
            sol.path_i[:s+1] = sol_shock.path_i[par.T-(s+1):]
            sol.path_w[:s+1] = sol_shock.path_w[par.T-(s+1):]

            for outputname in self.outputs_hh:
                varname = f'path_{outputname}'                     
                sol.__dict__[varname][:s+1] = sol_shock.__dict__[varname][par.T-(s+1):]

            # ii. after shock solution is ss
            sol.path_i[s+1:] = sol.i
            sol.path_w[s+1:] = sol.w
            par.z_grid_path[s+1:] = par.z_grid_ss
            par.z_trans_path[s+1:] = par.z_trans_ss

            for outputname in self.outputs_hh:
                varname = f'path_{outputname}'                     
                sol.__dict__[varname][s+1:] = sol.__dict__[outputname]

            # iii. simulate path
            self.simulate_hh_path()

            # iv. compute Jacobian
            for outputname in self.outputs_hh:
                
                jac_hh_ = jac_hh[(f'{outputname.upper()}_hh',inputname)]

                varname = f'path_{outputname}'
                for t in range(par.T):

                    basevalue = np.sum(sol.__dict__[outputname]*sim.D)
                    shockvalue = np.sum(sol.__dict__[varname][t]*sim.path_D[t])

                    jac_hh_[t,s] = (shockvalue-basevalue)/dx

        if do_print: print(f' [computed in {elapsed(t0)}]')

    def _calc_jac_hh_fakenews(self,jac_hh,inputname,dx=1e-4,do_print=False,do_print_full=False):
        """ compute Jacobian of household problem with fake news algorithm """
        
        par = self.par
        sol = self.sol
        sim = self.sim
        
        t0_all = time.time()
        
        if do_print or do_print_full: 
            print(f'inputname = {inputname}',end='')

        if do_print_full: 
            print('')
        elif do_print:
            print(': ',end='')
        
        # a. step 1: solve backwards
        t0 = time.time()
        
        self._set_inputs_hh_ss()

        if not inputname == 'ghost':
            shockarray = getattr(self.path,inputname)
            shockarray[0,-1] += dx

        self.solve_hh_path(do_print=False)
        self.sol_fakenews[inputname] = deepcopy(self.sol)

        if do_print_full: print(f'household problem solved backwards in {elapsed(t0)}')

        # b. step 2: derivatives
        t0 = time.time()
        
        diffs = self.diffs[inputname] = {}

        # allocate
        diffs['D'] = np.zeros((par.T,*sim.D.shape))
        for varname in self.outputs_hh: diffs[varname] = np.zeros(par.T)
        
        # compute
        D_ss = sim.D
        D_ini = np.zeros(D_ss.shape)     
        
        z_trans_ss_T = par.z_trans_ss.T
        z_trans_ss_T_inv = np.linalg.inv(z_trans_ss_T)  
        
        for s in range(par.T):
            
            t_ = (par.T-1) - s

            z_trans_T = par.z_trans_path[t_].T
            simulate_hh_D0(D_ss,z_trans_ss_T_inv,z_trans_T,D_ini)
            simulate_hh_forwards(D_ini,sol.path_i[t_],sol.path_w[t_],z_trans_ss_T,diffs['D'][s])
            
            diffs['D'][s] = (diffs['D'][s]-sim.D)/dx

            for outputname in self.outputs_hh:

                varname = f'path_{outputname}'

                basevalue = np.sum(sol.__dict__[outputname]*sim.D)
                shockvalue = np.sum(sol.__dict__[varname][t_]*D_ini)
                diffs[outputname][s] = (shockvalue-basevalue)/dx 
        
        if do_print_full: print(f'derivatives calculated in {elapsed(t0)}')
                        
        # c. step 3: expectation factors
        t0 = time.time()
        
        # demeaning improves numerical stability
        def demean(x):
            return x - x.sum()/x.size

        exp = self.exp[inputname] = {}

        for outputname in self.outputs_hh:

            sol_ss = sol.__dict__[outputname]

            exp[outputname] = np.zeros((par.T-1,*sol_ss.shape))
            exp[outputname][0] = demean(sol_ss)
       
        for t in range(1,par.T-1):
            
            for outputname in self.outputs_hh:
                simulate_hh_forwards_transpose(exp[outputname][t-1],sol.i,sol.w,par.z_trans_ss,exp[outputname][t])
                exp[outputname][t] = demean(exp[outputname][t])
            
        if do_print_full: print(f'expecation factors calculated in {elapsed(t0)}')
            
        # d. step 4: F        
        t0 = time.time()

        F = self.F[inputname] = {}
        for outputname in self.outputs_hh:
        
            F[outputname] = np.zeros((par.T,par.T))
            F[outputname][0,:] = diffs[outputname]
            F[outputname][1:, :] = exp[outputname].reshape((par.T-1, -1)) @ diffs['D'].reshape((par.T, -1)).T

        if do_print_full: print(f'f calculated in {elapsed(t0)}')
        
        t0 = time.time()
        
        # e. step 5: J
        J = {}

        for outputname in self.outputs_hh:
            J[outputname] = F[outputname].copy()
            for t in range(1, J[outputname].shape[1]): J[outputname][1:, t] += J[outputname][:-1, t - 1]

        if do_print_full: print(f'J calculated in {elapsed(t0)}')
            
        # f. save
        for outputname in self.outputs_hh:
            jac_hh[(f'{outputname.upper()}_hh',inputname)] = J[outputname]

        if do_print or do_print_full: print(f'household Jacobian computed in {elapsed(t0_all)}')
        if do_print_full: print('')
        
    def _compute_jac_hh(self,dx=1e-6,do_print=False,do_print_full=False,do_direct=False,s_list=None):
        """ compute Jacobian of household problem """

        t0 = time.time()

        path_original = deepcopy(self.path)
        if not do_direct: assert s_list is None, 'not implemented for fake news algorithm'

        self.sol_fakenews = {}
        self.diffs = {}
        self.exp = {}
        self.F = {}

        # a. ghost run
        jac_hh_ghost = {}
        if do_direct:
            self._calc_jac_hh_direct(jac_hh_ghost,'ghost',dx=dx,
                do_print=do_print,s_list=s_list)
        else:
            self._calc_jac_hh_fakenews(jac_hh_ghost,'ghost',dx=dx,
                do_print=do_print,do_print_full=do_print_full)

        # b. run for each input        
        jac_hh = {}
        for inputname in self.inputs_hh:
            if do_direct:
                self._calc_jac_hh_direct(jac_hh,inputname,dx=dx,
                    do_print=do_print,s_list=s_list)
            else:
                self._calc_jac_hh_fakenews(jac_hh,inputname,dx=dx,
                    do_print=do_print,do_print_full=do_print_full)

        # c. correction with ghost run
        for outputname in self.outputs_hh:
            for inputname in self.inputs_hh:
                
                # i. corrected value
                key = (f'{outputname.upper()}_hh',inputname)
                key_ghost = (f'{outputname.upper()}_hh','ghost')
                value = jac_hh[key]-jac_hh_ghost[key_ghost]
                
                # ii. dictionary
                self.jac_hh[key] = value

        if do_print: print(f'all Jacobians computed in {elapsed(t0)}')

        # reset
        self.path = path_original

    def _compute_jac(self,do_shocks=False,dx=1e-6,do_print=False,parallel=True):
        """ compute full Jacobian """
        
        do_unknowns = not do_shocks

        t0 = time.time()
        evaluate_t = 0.0

        path_original = deepcopy(self.path)

        if do_unknowns:
            inputs = self.unknowns
        else:
            inputs = self.shocks

        par = self.par
        path = self.path

        # a. set shocks and unknowns to steady state
        self._set_shocks_ss()
        self._set_unknowns_ss()
        
        # b. baseline evaluation at steady state
        t0_ = time.time()
        self.evaluate_path(use_jac_hh=True)
        evaluate_t += time.time()-t0_

        base = self._get_errors().ravel() 
        path_ss = deepcopy(path)

        # c. calculate
        if do_unknowns:
            jac_mat = self.H_U
        else:
            jac_mat = self.H_Z
        
        x_ss = np.zeros((len(inputs),par.T))
        for i,varname in enumerate(inputs):
            x_ss[i,:] = getattr(self.ss,varname)

        if parallel:
            
            # i. inputs
            x0 = np.zeros((x_ss.size,x_ss.size))
            for i in range(x_ss.size):   

                x0[:,i] = x_ss.ravel().copy()
                x0[i,i] += dx

            # ii. evaluate
            self._set_unknowns(x0,inputs,parallel=True)
            t0_ = time.time()
            self.evaluate_path(threads=len(inputs)*par.T,use_jac_hh=True)
            evaluate_t += time.time()-t0_
            errors = self._get_errors(inputs,parallel=True)

            # iii. Jacobian
            jac_mat[:,:] = (errors.reshape(jac_mat.shape)-base[:,np.newaxis])/dx

            # iv. all other variables
            for i_input,inputname in enumerate(inputs):
                for outputname in self.varlist:
                    
                    key = (outputname,inputname)
                    jac = self.jac[key]

                    for s in range(par.T):

                        thread = i_input*par.T+s
                        jac[:,s] = (path.__dict__[outputname][thread,:]-path_ss.__dict__[outputname][0,:])/dx

        else:

            for i in range(x_ss.size):   
                
                # i. inputs
                x0 = x_ss.ravel().copy()
                x0[i] += dx
                
                # ii. evaluations
                self._set_unknowns(x0,inputs)

                t0_ = time.time()
                self.evaluate_path(use_jac_hh=True)
                evaluate_t += time.time()-t0_

                errors = self._get_errors() 
                                
                # iii. Jacobian
                jac[:,i] = (errors.ravel()-base)/dx
        
        if do_print:
            if do_unknowns:
                print(f'full Jacobian to unknowns computed in {elapsed(t0)} [in evaluate_path(): {elapsed(0,evaluate_t)}]')
            else: 
                print(f'full Jacobian to shocks computed in {elapsed(t0)} [in evaluate_path(): {elapsed(0,evaluate_t)}]')

        # reset
        self.path = path_original
 
    def compute_jacs(self,dx=1e-6,skip_hh=False,skip_shocks=False,do_print=False):
        """ compute all Jacobians """
        
        if not skip_hh:
            if do_print: print('household Jacobians:')
            self._compute_jac_hh(dx=dx,do_print=do_print)
            if do_print: print('')

        if do_print: print('full Jacobians:')
        self._compute_jac(dx=dx,do_print=do_print)
        if not skip_shocks: self._compute_jac(do_shocks=True,dx=dx,do_print=do_print)

    ####################################
    # 5. find transition path and IRFs #
    ####################################

    def _set_shocks(self,shock_specs=None,std_shock=False):
        """ set shocks based on shock specification, default is AR(1) """
        
        if shock_specs is None: shock_specs = {}

        for shockname in self.shocks:

            patharray = getattr(self.path,shockname)
            ssvalue = getattr(self.ss,shockname)
            
            # a. custom path
            if (dshockname := f'd{shockname}') in shock_specs:
                patharray[:,:] = ssvalue + shock_specs[dshockname]

            # b. AR(1) path
            else:

                # i. jump and rho
                if std_shock:
                    
                    stdname = f'std_{shockname}'
                    scale = getattr(self.par,stdname)

                    assert not scale < 0, f'{stdname} must not be negative'

                else:

                    jumpname = f'jump_{shockname}'
                    scale = getattr(self.par,jumpname)
                
                rhoname = f'rho_{shockname}'
                rho = getattr(self.par,rhoname)

                # ii. set value
                patharray[:,:] = ssvalue +  scale*rho**np.arange(self.par.T)

    def find_IRFs(self,shock_specs=None,reuse_G_U=False,do_print=False):
        """ find linearized impulse responses """
        
        par = self.par
        ss = self.ss
        path = self.path

        t0 = time.time()

        # a. solution matrix
        t0_ = time.time()
        if not reuse_G_U: self.G_U[:,:] = -np.linalg.solve(self.H_U,self.H_Z)       
        t1_ = time.time()
        
        # b. set path for shocks
        self._set_shocks(shock_specs=shock_specs)

        # c. IRFs

        # shocks
        dZ = np.zeros((len(self.shocks),par.T))
        for i_shock,shockname in enumerate(self.shocks):
            dZ[i_shock,:] = path.__dict__[shockname][0,:]-ss.__dict__[shockname]
            self.IRF[shockname][:] = dZ[i_shock,:] 

        # unknowns
        dU = self.G_U@dZ.ravel()
        dU = dU.reshape((len(self.unknowns),par.T))

        for i_unknown,unknownname in enumerate(self.unknowns):
            self.IRF[unknownname][:] =  dU[i_unknown,:]

        # remaing
        for varname in self.varlist:

            if varname in self.shocks+self.unknowns: continue

            self.IRF[varname][:] = 0.0
            for inputname in self.shocks+self.unknowns:
                self.IRF[varname][:] += self.jac[(varname,inputname)]@self.IRF[inputname]

        if do_print: print(f'linear transition path found in {elapsed(t0)} [finding solution matrix: {elapsed(t0_,t1_)}]')

    def evaluate_path(self,threads=1,use_jac_hh=False):
        """ evaluate transition path """

        sol = self.sol
        sim = self.sim
        ss = self.ss
        path = self.path

        assert use_jac_hh or threads == 1
        
        # a. before household block
        with jit(self) as model:
            self.block_pre(model.par,model.sol,model.sim,model.ss,model.path,threads=threads)

        # b. household block
        if use_jac_hh and len(self.outputs_hh) > 0: # linearized

            for outputname in self.outputs_hh:
                
                Outputname_hh = f'{outputname.upper()}_hh'

                # i. set steady state value
                pathvalue = path.__dict__[Outputname_hh]
                ssvalue = ss.__dict__[Outputname_hh]
                pathvalue[:,:] = ssvalue

                # ii. update with Jacobians and inputs
                for inputname in self.inputs_hh:

                    jac_hh = self.jac_hh[(f'{Outputname_hh}',inputname)]

                    ssvalue_input = ss.__dict__[inputname]
                    pathvalue_input = path.__dict__[inputname]

                    pathvalue[:,:] += (jac_hh@(pathvalue_input.T-ssvalue_input)).T 
                    # transposing needed for correct broadcasting
        
        elif len(self.outputs_hh) > 0: # non-linear solution

            # i. solve
            self.solve_hh_path()

            # ii. simulate
            self.simulate_hh_path()

            # iii. aggregate
            for outputname in self.outputs_hh:

                Outputname_hh = f'{outputname.upper()}_hh'
                pathvalue = path.__dict__[Outputname_hh]

                pol = sol.__dict__[f'path_{outputname}']
                pathvalue[:] = np.sum(pol*sim.path_D,axis=tuple(range(1,pol.ndim)))
                # sum over all but first dimension

        else:

            pass # no household block
                
        # c. after household block
        with jit(self) as model:
            self.block_post(model.par,model.sol,model.sim,model.ss,model.path,threads=threads)

    def _evaluate_H(self,x,do_print=False):
        """ compute error in equation system for targets """
        
        par = self.par

        # a. evaluate
        self._set_unknowns(x,self.unknowns)
        self.evaluate_path()
        errors = self._get_errors() 
        
        # b. print
        if do_print: 
            
            max_abs_error = np.max(np.abs(errors))

            for k in self.targets:
                v = getattr(self.path,k)
                print(f'{k:10s} = {np.max(np.abs(v)):8.1e}')

            print(f'\nmax abs. error: {max_abs_error:8.1e}')

        # c. return as vector
        return errors.ravel()

    def find_transition_path(self,shock_specs=None,do_print=False):
        """ find transiton path (fully non-linear) """

        par = self.par

        t0 = time.time()

        # a. set path for shocks
        self._set_shocks(shock_specs=shock_specs)

        # b. set initial value of endogenous inputs to ss
        x0 = np.zeros((len(self.unknowns),par.T))
        for i,varname in enumerate(self.unknowns):
            x0[i,:] = getattr(self.ss,varname)

        # c. solve
        obj = lambda x: self._evaluate_H(x)

        if do_print: print(f'finding the transition path:')
        x = broyden_solver(obj,x0,self.H_U,
            tol=par.tol_broyden,
            max_iter=par.max_iter_broyden,
            targets=self.targets,
            do_print=do_print)
        
        # d. final evaluation
        self._evaluate_H(x)

        if do_print: print(f'\ntransition path found in {elapsed(t0)}')

    ###########
    # 6. IRFs #
    ###########

    def show_IRFs(self,varnames,
        abs_diff=None,lvl_value=None,facs=None,pows=None,
        do_shocks=True,do_targets=True,do_linear=False,
        T_max=None,ncols=4,filename=None):
        """ shows IRFS """

        # varnames: list[str], variable names
        # abs_diff: list[str], variable names to be shown as absolute difference to ss (defaulte is in % if ss is not nan)
        # lvl_value: list[str], variable names to be shown level (defaulte is in % if ss is not nan)
        # facs: dict[str -> float], scaling factor when in abs_diff or lvl_value
        # pows: dict[str -> float], scaling power when in abs_diff or lvl_value
        # do_shocks: boolean, show IRFs for the inputs
        # do_targets: boolean, show IRFs for the targets
        # T_max: int, length of IRF
        # ncols: number of columns
        # filename: filename if saving figure

        models = [self]
        labels = ['non-linear']
        show_IRFs(models,labels,varnames,
            abs_diff=abs_diff,lvl_value=lvl_value,facs=facs,pows=pows,
            do_shocks=do_shocks,do_targets=do_targets,do_linear=do_linear,
            T_max=T_max,ncols=ncols,filename=filename)

    def compare_IRFs(self,models,labels,varnames,
        abs_diff=None,lvl_value=None,facs=None,pows=None,
        do_shocks=True,do_targets=True,
        T_max=None,ncols=4,filename=None):
        """ compare IRFs across models """

        # models: list[GEModelClass], models
        # labels: list[str], model names
        # varnames: list[str], variable names
        # abs_diff: list[str], variable names to be shown as absolute difference to ss (defaulte is in % if ss is not nan)
        # lvl_value: list[str], variable names to be shown level (defaulte is in % if ss is not nan)
        # facs: dict[str -> float], scaling factor when in abs_diff or lvl_value
        # pows: dict[str -> float], scaling power when in abs_diff or lvl_value
        # do_shocks: boolean, show IRFs for the inputs
        # do_targets: boolean, show IRFs for the targets
        # T_max: int, length of IRF
        # ncols: number of columns
        # filename: filename if saving figure
                 
        show_IRFs(models,labels,varnames,
            abs_diff=abs_diff,lvl_value=lvl_value,facs=facs,pows=pows,
            do_shocks=do_shocks,do_targets=do_targets,
            T_max=T_max,ncols=ncols,filename=filename)
  
    ###############
    # 7. simulate #
    ###############

    def prepare_simulate(self,reuse_G_U=False,do_print=True):
        """ prepare model for simulation by calculating IRFs """

        par = self.par
        ss = self.ss
        sol = self.sol
        sol_fakenews = self.sol_fakenews
        path = self.path

        t0 = time.time()

        # a. solution matrix
        t0_sol = time.time()
        if not reuse_G_U: self.G_U[:,:] = -np.linalg.solve(self.H_U,self.H_Z)       
        t1_sol = time.time()   
        
        # b. calculate unit shocks
        self._set_shocks(std_shock=True)

        # c. IRFs
        for i_input,shockname in enumerate(self.shocks):
            
            # i. shock
            dZ = np.zeros((len(self.shocks),par.T))        
            dZ[i_input,:] = path.__dict__[shockname][0,:]-ss.__dict__[shockname]

            self.IRF[(shockname,shockname)][:] = dZ[i_input,:] 

            # ii. unknowns
            dU = self.G_U@dZ.ravel()
            dU = dU.reshape((len(self.unknowns),par.T))

            for i_unknown,unknownname in enumerate(self.unknowns):                
                self.IRF[(unknownname,shockname)][:] = dU[i_unknown,:]

            # iii. remaining
            for varname in self.varlist:

                if varname in self.shocks+self.unknowns: continue

                self.IRF[(varname,shockname)][:] = 0.0
                for inputname in self.shocks+self.unknowns:
                    self.IRF[(varname,shockname)][:] += self.jac[(varname,inputname)]@self.IRF[(inputname,shockname)]

        # d. household
        t0_hh = time.time()

        dpols = {}
        for polname in self.pols_hh:
            base = getattr(sol_fakenews['ghost'],f'path_{polname}')
            for inputname in self.inputs_hh:    
                value = getattr(sol_fakenews[inputname],f'path_{polname}')
                dpols[(polname,inputname)] = np.flip((value-base)/1e-6,axis=0)

        self.IRF['pols'] = {}
        for shockname in self.shocks:  
            for polname in self.pols_hh:
                IRF_pols = self.IRF['pols'][(polname,shockname)] = np.zeros((*sol.i.shape,par.T))
                for inputname in self.inputs_hh:    
                    update_IRF_hh(IRF_pols,dpols[(polname,inputname)],self.IRF[(inputname,shockname)])
                    
        t1_hh = time.time()

        if do_print: print(f'simulation prepared in {elapsed(t0)} [solution matrix: {elapsed(t0_sol,t1_sol)}, household: {elapsed(t0_hh,t1_hh)}]')

    def simulate(self,do_prepare=True,reuse_G_U=False,do_print=False):
        """ simulate the model """

        par = self.par
        sim = self.sim
        sol = self.sol
        path = self.path
        
        # a. prepare simulation
        if do_prepare: self.prepare_simulate(reuse_G_U=reuse_G_U,do_print=do_print)

        t0 = time.time()

        # a. IRF matrix
        IRF_mat = np.zeros((len(self.varlist),len(self.shocks),par.T))
        for i,varname in enumerate(self.varlist):
            for j,shockname in enumerate(self.shocks):
                IRF_mat[i,j,:] = self.IRF[(varname,shockname)]

        # b. draw shocks
        epsilons = np.random.normal(size=(len(self.shocks),par.simT))

        # c. simulate
        sim_mat = np.zeros((len(self.varlist),par.simT))
        simulate_agg(epsilons,IRF_mat,sim_mat)

        for i,varname in enumerate(self.varlist):
            sim.__dict__[f'd{varname}'][:] = sim_mat[i]

        if do_print: print(f'aggregates simulated in {elapsed(t0)}')

        # d. households
        self.sim_alt = {}

        t0 = time.time()

        # i. policies
        IRF_pols_mat = np.zeros((len(self.pols_hh),len(self.shocks),*sol.i.shape,par.T))
        for i,polname in enumerate(self.pols_hh):
            for j,shockname in enumerate(self.shocks):
                IRF_pols_mat[i,j] = self.IRF['pols'][(polname,shockname)]

        sim_pols_mat = self.sim_alt['pols'] = np.zeros((len(self.pols_hh),par.simT,*sol.i.shape))
        
        t0_ = time.time()
        simulate_agg_hh(epsilons,IRF_pols_mat,sim_pols_mat)
        t1_ = time.time()

        for i,polname in enumerate(self.pols_hh):
            sim_pols_mat[i] += getattr(sol,polname)

        if do_print: print(f'household policies simulated in {elapsed(t0)} [in aggregation: {elapsed(t0_,t1_)}]')     

        # ii. distribution
        t0 = time.time()

        sim_D = self.sim_alt['D'] = np.zeros((par.simT,*sim.D.shape))
        sim_i = np.zeros(sol.i.shape,dtype=np.int_)
        sim_w = np.zeros(sol.w.shape)

        z_trans_T = par.z_trans_ss.T
        z_trans_T_inv = np.linalg.inv(z_trans_T)

        if len(self.grids_hh) == 1:

            grid1 = getattr(par,f'{self.grids_hh[0]}_grid')

            for t in range(par.simT):
            
                if t == 0:
                    simulate_hh_D0(sim.D,z_trans_T_inv,z_trans_T,sim_D[t])    
                else:
                    find_i_and_w_1d_1d(sim_pols_mat[0,t],grid1,sim_i,sim_w)
                    simulate_hh_forwards(sim_D[t-1],sim_i,sim_w,z_trans_T,sim_D[t])

        else:

            raise NotImplementedError

        if do_print: print(f'distribution simulated in {elapsed(t0)}')     

        # iii. aggregate
        t0 = time.time()

        for i,polname in enumerate(self.pols_hh):

            Outputname_hh = f'd{polname.upper()}_hh'
            self.sim_alt[Outputname_hh] = np.sum(sim_pols_mat[0,i]*sim_D,axis=tuple(range(1,sim_pols_mat[0,i].ndim+1)))
            # sum over all but first dimension
            
        if do_print: print(f'aggregates calculated from distribution {elapsed(t0)}')       
        
    ############
    # 8. tests #
    ############

    test_hh_path = tests.hh_path
    test_path = tests.path
    test_jac_hh = tests.jac_hh