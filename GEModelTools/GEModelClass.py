# contains main GEModelClass

import time
from types import SimpleNamespace
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams.update({'font.size':12})

from consav import jit
from consav.misc import elapsed

from .simulate import find_i_and_w_1d_1d, find_i_and_w_1d_1d_path
from .simulate import simulate_hh_initial_distribution, simulate_hh_forwards, simulate_hh_forwards_transpose
from .simulate import simulate_hh_path, simulate_hh_ss
from .broyden_solver import broyden_solver
from .figures import show_IRFs

import grids
import household_problem
import steady_state
import transition_path

class GEModelClass:

    ############
    # 1. setup #
    ############

    def allocate_GE(self,sol_shape):
        """ allocate GE variables """

        par = self.par
        sol = self.sol
        sim = self.sim

        _Nfix = sol_shape[0]
        Nz = sol_shape[1]
        Nendos = sol_shape[2:]

        path_sol_shape = (par.transition_T,*sol_shape)
        
        # a. defaults
        par.__dict__.setdefault('transition_T',500)
        par.__dict__.setdefault('max_iter_solve',50_000)
        par.__dict__.setdefault('max_iter_simulate',50_000)
        par.__dict__.setdefault('max_iter_broyden',100)
        par.__dict__.setdefault('tol_solve',1e-10)
        par.__dict__.setdefault('tol_simulate',1e-10)
        par.__dict__.setdefault('tol_broyden',1e-8)

        for varname in self.inputs_exo:
            par.__dict__.setdefault(f'jump_{varname}',0.0)
            par.__dict__.setdefault(f'rho_{varname}',0.0)

        assert hasattr(self.par,'Nz'), 'par.Nz must be specified'

        # automatic not-floats
        not_floats = ['transition_T','max_iter_solve','max_iter_simulate','max_iter_broyden'] 
        not_floats += [f'N{varname}' for varname in self.grids_hh] + ['Nz']

        for not_float in not_floats:
            if not not_float in self.not_floats:
                self.not_floats.append(not_float)

        # b. checks
        assert Nz == par.Nz, f'sol_shape is wrong, sol_shape[1] = {Nz}, par.Nz = {par.Nz}'

        attrs = ['grids_hh','pols_hh','inputs_hh','inputs_exo','inputs_endo','targets','varlist_hh','varlist','jac']
        attrs += ['par','sol','sim','ss','path','jac_hh']
        for attr in attrs:
            assert hasattr(self,attr), f'missing .{attr}'

        for i,(varname,Nendo) in enumerate(zip(self.grids_hh,Nendos)):
            Nx = f'N{varname}'
            assert hasattr(par,Nx), f'{Nx} not in .par'
            Nxval = getattr(par,Nx)
            assert Nendo == Nxval, f'sol_shape is wrong, sol_shape[{i}] = {Nendo}, par.{Nx} = {Nxval}'
        
        for varname in self.grids_hh + self.pols_hh + self.outputs_hh:
            assert varname in self.varlist_hh, f'{varname} not in .varlist_hh'
        
        for varname in self.inputs_hh:
            assert varname in self.varlist, f'{varname} not in .varlist'

        for varname in self.outputs_hh:
            varname_agg = varname.upper()
            assert varname_agg in self.varlist, f'{varname_agg} not in .varlist'
            hasattr(self.path,varname_agg), f'{varname_agg} not in .path'

        for varname in self.inputs_exo + self.inputs_endo + self.targets:
            assert varname in self.varlist, f'{varname} not in .varlist'

        # c. allocate grids and transition matrices
        for varname in self.grids_hh:

            gridname = f'{varname}_grid' 
            Nx = getattr(par,f'N{varname}')
            gridarray = np.zeros(Nx)
            setattr(par,gridname,gridarray)
            
        par.z_grid_ss = np.zeros(par.Nz)
        par.z_trans_ss = np.zeros((par.Nz,par.Nz))
        par.z_ergodic_ss = np.zeros(par.Nz)
        
        par.z_grid_path = np.zeros((par.transition_T,par.Nz))
        par.z_trans_path = np.zeros((par.transition_T,par.Nz,par.Nz))        

        # d. allocate household variables
        for varname in self.varlist_hh:

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
        path_shape = (len(self.inputs_endo)*par.transition_T,par.transition_T)
        for varname in self.varlist:
            setattr(self.ss,varname,np.nan)
            setattr(self.path,varname,np.zeros(path_shape))

        # g. allocate household Jacobians
        jac_hh = self.jac_hh
        for inname in self.inputs_hh:
            for outname in self.outputs_hh:
                jacarray = np.zeros((par.transition_T,par.transition_T))
                setattr(jac_hh,f'{outname.upper()}_{inname}',jacarray)

    def create_grids(self):
        """ create grids """

        grids.create_grids(self)

    def print_unpack_varlist(self):
        """ print varlist for use in evaluate_path() """

        print(f'    for thread in nb.prange(threads):\n')
        print('        # unpack')
        for varname in self.varlist:
            print(f'        {varname} = path.{varname}[thread,:]')

    ####################
    # 2. steady state #
    ###################

    def solve_hh_ss(self,do_print=False):
        """ solve the household problem in steady state """

        t0 = time.time()

        # a. create (or re-create) grids
        self.create_grids()

        # b. solve
        with jit(self) as model:
            
            par = model.par
            sol = model.sol
            ss = model.ss

            it = household_problem.solve_hh_ss(par,sol,ss)
        
            if do_print:
                print(f'household problem in ss solved in {elapsed(t0)} [{it} iterations]')

        # d. indices and weights    
        par = model.par
        sol = model.sol
        ss = model.ss

        if len(self.grids_hh) == 1 and len(self.pols_hh) == 1:
            pol1 = getattr(sol,f'{self.pols_hh[0]}')
            grid1 = getattr(par,f'{self.grids_hh[0]}_grid') 
            find_i_and_w_1d_1d(pol1,grid1,sol.i,sol.w)
        else:
            raise ValueError('not implemented')

    def simulate_hh_ss(self,do_print=False):
        """ simulate the household problem in steady state """
        
        par = self.par
        sol = self.sol
        sim = self.sim        
        t0 = time.time()

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
                raise ValueError('not implemented')
    
        # b. simulate
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim
            
            it = simulate_hh_ss(par,sol,sim)

        if do_print:
            print(f'household problem in ss simulated in {elapsed(t0)} [{it} iterations]')

    def find_ss(self,do_print=False):
        """ solve for the model steady state """

        steady_state.find_ss(self,do_print=do_print)

    #####################
    # 3. household path #
    #####################

    def solve_hh_path(self,do_print=False):
        """ gateway for solving the household problem along the transition path """

        t0 = time.time()

        # a. solve
        with jit(self) as model:
            
            household_problem.solve_hh_path(model.par,model.sol,model.ss,model.path)

        # b. indices and weights
        par = self.par
        sol = self.sol
        ss = self.ss

        if len(self.grids_hh) == 1 and len(self.pols_hh) == 1:
            path_pol1 = getattr(sol,f'path_{self.pols_hh[0]}')
            grid1 = getattr(par,f'{self.grids_hh[0]}_grid') 
            find_i_and_w_1d_1d_path(par.transition_T,path_pol1,grid1,sol.path_i,sol.path_w)
        else:
            raise ValueError('not implemented')

        if do_print:
            print(f'household problem solved along transition path in {elapsed(t0)}')

    def simulate_hh_path(self,do_print=False):
        """ gateway for simulating the household problem along the transition path"""
        
        t0 = time.time()

        par = self.par
        sim = self.sim
        sol = self.sol

        with jit(self) as model:
            simulate_hh_path(model.par,model.sol,model.sim)

        if do_print:
            print(f'household problem simulated along transition in {elapsed(t0)}')

    def _set_inputs_hh_ss(self):
        """ set household inputs to steady state """

        for inputname in self.inputs_hh:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    def test_hh_path(self,d=1e-4):
        """ test household solution and simulation along path """

        print('note: inputs = steady state value -> expected: constant value (straigt line)')

        par = self.par
        sim = self.sim
        sol = self.sol
        ss = self.ss

        # a. solution and simulation hh along path
        self._set_inputs_hh_ss()
        self.solve_hh_path(do_print=True)
        self.simulate_hh_path(do_print=True)

        # b. show mean of each hh output
        fig = plt.figure(figsize=(6,len(self.outputs_hh)*4))
        for i,outputname in enumerate(self.outputs_hh):
            
            D = sim.path_D
            pol = getattr(sol,f'path_{outputname}')
            y = [np.sum(D[t]*pol[t])for t in range(par.transition_T)]

            ax = fig.add_subplot(len(self.outputs_hh),1,1+i)
            ax.plot(np.arange(par.transition_T),y,'-')
            ax.set_title(outputname)

            ssvalue = getattr(ss,f'{outputname.upper()}_hh')
            ax.set_ylim([(1-d)*ssvalue,(1+d)*ssvalue])

    ################
    # 4. Jacobians #
    ################

    def _set_inputs_exo_ss(self):
        """ set endogenous  inputs to steady state """

        for inputname in self.inputs_exo:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    def _set_inputs_endo_ss(self):
        """ set endogenous  inputs to steady state """

        for inputname in self.inputs_endo:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    def test_path(self):
        """ test evaluation of path """

        print('note: inputs = steady state value -> expected: no difference to steady state and zero errors\n')

        par = self.par
        ss = self.ss
        path = self.path

        # a. set exogenous and endogenous to steady state
        self._set_inputs_exo_ss()
        self._set_inputs_endo_ss()
        
        # b. baseline evaluation at steady state 
        self.evaluate_path()

        # c. 
        print('difference to value at steady state:')
        for varname in self.varlist:

            pathvalue = getattr(path,varname)[0,:]
            ssvalue = getattr(ss,varname)

            if np.isnan(ssvalue): continue

            diff_t0 = pathvalue[0]-ssvalue
            max_abs_diff = np.max(np.abs(pathvalue-ssvalue))

            print(f'{varname:15s}: t0 = {diff_t0:12.8f}, max abs. {max_abs_diff:12.8f}')

        print('\nabsolute value (potential targets):')
        for varname in self.varlist:

            pathvalue = getattr(path,varname)[0,:]
            ssvalue = getattr(ss,varname)

            if not np.isnan(ssvalue): continue
            
            max_abs = np.max(np.abs(pathvalue))

            print(f'{varname:15s}: t0 = {pathvalue[0]:12.8f}, max abs. {max_abs:12.8f}')

    def _calc_jac_hh_direct(self,shockname,dshock=1e-4,do_print=False,jac_hh=None,s_list=None):
        """ compute jacobian of household problem """

        par = self.par
        sol = self.sol
        sim = self.sim
        jac_hh = self.jac_hh if jac_hh is None else jac_hh
        
        if s_list is None: s_list = list(range(par.transition_T))

        t0 = time.time()
        if do_print: print(f'finding jacobian wrt. {shockname:3s}:',end='')
            
        # a. allocate
        for outputname in self.outputs_hh:
            jacarray = np.zeros((par.transition_T,par.transition_T))
            setattr(jac_hh,f'{outputname.upper()}_{shockname}',jacarray)

        # b. solve with shock in last period
        self._set_inputs_hh_ss()

        if not shockname == '__ghost':
            shockarray = getattr(self.path,shockname)
            shockarray[0,-1] += dshock

        self.solve_hh_path()
        par_shock = deepcopy(self.par)
        sol_shock = deepcopy(self.sol)

        # c. simulate
        for s in s_list:

            if do_print: print(f' {s}',end='')
            
            # i. before shock only time to shock matters
            par.z_grid_path[:s+1] = par_shock.z_grid_path[par.transition_T-(s+1):]
            par.z_trans_path[:s+1] = par_shock.z_trans_path[par.transition_T-(s+1):]
            sol.path_i[:s+1] = sol_shock.path_i[par.transition_T-(s+1):]
            sol.path_w[:s+1] = sol_shock.path_w[par.transition_T-(s+1):]

            for outputname in self.outputs_hh:
                varname = f'path_{outputname}'                     
                sol.__dict__[varname][:s+1] = sol_shock.__dict__[varname][par.transition_T-(s+1):]

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

            # iv. compute jacobian
            for outputname in self.outputs_hh:
                
                jacarray = getattr(jac_hh,f'{outputname.upper()}_{shockname}')
                varname = f'path_{outputname}'

                for t in range(par.transition_T):

                    basevalue = np.sum(sol.__dict__[outputname]*sim.D)
                    shockvalue = np.sum(sol.__dict__[varname][t]*sim.path_D[t])

                    jacarray[t,s] = (shockvalue-basevalue)/dshock

        if do_print: print(f' [computed in {elapsed(t0)}]')

    def _calc_jac_hh_fakenews(self,shockname,dshock=1e-4,do_print=False,do_print_full=False,jac_hh=None):
        """ compute jacobian of household problem with fake news algorithm """
        
        par = self.par
        sol = self.sol
        sim = self.sim
        jac_hh = self.jac_hh if jac_hh is None else jac_hh

        t0_all = time.time()
        
        if do_print: print(f'shockname = {shockname}',end='')
        if do_print_full:
            print('')
        else:
            print(': ',end='')
        
        # a. step 1: solve backwards
        t0 = time.time()
        
        self._set_inputs_hh_ss()

        if not shockname == '__ghost':
            shockarray = getattr(self.path,shockname)
            shockarray[0,-1] += dshock

        self.solve_hh_path(do_print=False)

        if do_print_full: print(f'household problem solved backwards in {elapsed(t0)}')

        # b. step 2: derivatives
        t0 = time.time()
        
        diffs = {}

        # allocate
        diffs['D'] = np.zeros((par.transition_T,*sim.D.shape))
        for varname in self.outputs_hh: diffs[varname] = np.zeros(par.transition_T)
        
        # compute
        D_ss = sim.D
        D_ini = sim.D.copy()      
        
        z_trans_ss_T = par.z_trans_ss.T
        z_trans_ss_T_inv = np.linalg.inv(z_trans_ss_T)  
        
        for s in range(par.transition_T):
            
            t_ = (par.transition_T-1) - s

            z_trans_T = par.z_trans_path[t_].T
            simulate_hh_initial_distribution(D_ss,z_trans_ss_T_inv,z_trans_T,D_ini)
            simulate_hh_forwards(D_ini,sol.path_i[t_],sol.path_w[t_],z_trans_ss_T,diffs['D'][s])
            
            diffs['D'][s] = (diffs['D'][s]-sim.D)/dshock

            for outputname in self.outputs_hh:

                varname = f'path_{outputname}'

                basevalue = np.sum(sol.__dict__[outputname]*sim.D)
                shockvalue = np.sum(sol.__dict__[varname][t_]*D_ini)
                diffs[outputname][s] = (shockvalue-basevalue)/dshock 
        
        if do_print_full: print(f'derivatives calculated in {elapsed(t0)}')
                        
        # c. step 3: expectation factors
        t0 = time.time()
        
        # demeaning improves numerical stability
        def demean(x):
            return x - x.sum()/x.size

        exp = {}

        for outputname in self.outputs_hh:

            sol_ss = sol.__dict__[outputname]

            exp[outputname] = np.zeros((par.transition_T-1,*sol_ss.shape))
            exp[outputname][0] = demean(sol_ss)
       
        for t in range(1,par.transition_T-1):
            
            for outputname in self.outputs_hh:
                simulate_hh_forwards_transpose(exp[outputname][t-1],sol.i,sol.w,par.z_trans_ss,exp[outputname][t])
                exp[outputname][t] = demean(exp[outputname][t])
            
        if do_print_full: print(f'expecation factors calculated in {elapsed(t0)}')
            
        # d. step 4: F        
        t0 = time.time()

        F = {}
        for outputname in self.outputs_hh:
        
            F[outputname] = np.zeros((par.transition_T,par.transition_T))
            F[outputname][0,:] = diffs[outputname]
            F[outputname][1:, :] = exp[outputname].reshape((par.transition_T-1, -1)) @ diffs['D'].reshape((par.transition_T, -1)).T

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
            setattr(jac_hh,f'{outputname.upper()}_{shockname}',J[outputname])

        if do_print: print(f'household Jacobian computed in {elapsed(t0_all)}')
        if do_print_full: print('')
        
    def compute_jac_hh(self,dshock=1e-6,do_print=False,do_print_full=False,do_direct=False,s_list=None):
        """ compute jacobian of household problem """

        path_original = deepcopy(self.path)

        if not do_direct:
            assert s_list is None, 'not implemented'

        # a. ghost run
        jac_hh_ghost = SimpleNamespace()
        if do_direct:
            self._calc_jac_hh_direct('__ghost',dshock=dshock,do_print=do_print,jac_hh=jac_hh_ghost,s_list=s_list)
        else:
            self._calc_jac_hh_fakenews('__ghost',dshock=dshock,do_print=do_print,do_print_full=do_print_full,jac_hh=jac_hh_ghost)

        # b. run for each input
        for shockname in self.inputs_hh:
            if do_direct:
                self._calc_jac_hh_direct(shockname,dshock=dshock,do_print=do_print,s_list=s_list)
            else:
                self._calc_jac_hh_fakenews(shockname,dshock=dshock,do_print=do_print,do_print_full=do_print_full)

        # c. correction with ghost run
        for varname in self.outputs_hh:
            for shockname in self.inputs_hh:
                
                name = f'{varname.upper()}_{shockname}'
                name_ghost = f'{varname.upper()}___ghost'
                
                jacarray = getattr(self.jac_hh,name)
                jacarray_ghost = getattr(jac_hh_ghost,name_ghost)

                setattr(self.jac_hh,name,jacarray-jacarray_ghost)

        # reset
        self.path = path_original

    def test_jac_hh(self,s_list,dshock=1e-6):
        """ test the computation of hh Jacobians with direct and fake news method"""

        par = self.par

        print('note: differences should only be due to numerical errors\n')

        # a. direct
        print('direct method:')
        self.compute_jac_hh(dshock=dshock,do_print=True,do_print_full=False,do_direct=True,s_list=s_list)
        jac_hh_direct = deepcopy(self.jac_hh)

        # b. fake news
        print(f'\nfake news method:')
        self.compute_jac_hh(dshock=dshock,do_print=True,do_print_full=False,do_direct=False)

        # c. compare
        fig = plt.figure(figsize=(6*2,len(self.outputs_hh)*len(self.inputs_hh)*4),dpi=100)

        i = 0
        for inputname in self.inputs_hh:
            for outputname in self.outputs_hh:
            
                jac_hh_var_direct = getattr(jac_hh_direct,f'{outputname.upper()}_{inputname}')
                jac_hh_var = getattr(self.jac_hh,f'{outputname.upper()}_{inputname}')
                
                ax = fig.add_subplot(len(self.inputs_hh)*len(self.outputs_hh),2,i*2+1)
                ax_diff = fig.add_subplot(len(self.inputs_hh)*len(self.outputs_hh),2,i*2+2)

                ax.set_title(f'{outputname.upper()} to {inputname}')
                ax_diff.set_title(f'... difference')

                for j,s in enumerate(s_list):
                    
                    ax.plot(np.arange(par.transition_T),jac_hh_var_direct[:,s],color=colors[j],label=f'shock at {s}')
                    ax.plot(np.arange(par.transition_T),jac_hh_var[:,s],color=colors[j],ls='--',label='fake news')
                    
                    diff = jac_hh_var[:,s]-jac_hh_var_direct[:,s]
                    ax_diff.plot(np.arange(par.transition_T),diff,color=colors[j])

                if i == 0: ax.legend(frameon=True)
                i += 1    

    def compute_jac(self,h=1e-4,do_print=False,parallel=True):
        """ compute full jacobian """
        
        t0 = time.time()
        
        path_original = deepcopy(self.path)

        par = self.par
        ss = self.ss
        path = self.path

        x_shape = (len(self.inputs_endo),par.transition_T)
        
        # a. set exogenous and endogenous to steady state
        self._set_inputs_exo_ss()
        self._set_inputs_endo_ss()
        
        # b. baseline evaluation at steady state 
        x_ss = np.zeros(x_shape)
        for i,varname in enumerate(self.inputs_endo):
            x_ss[i,:] = getattr(self.ss,varname)
        
        base = self._path_obj(x_ss,use_jac_hh=True)
        
        # c. calculate
        jac = self.jac = np.zeros((x_ss.size,x_ss.size))

        if parallel:
            
            x0 = np.zeros((x_ss.size,x_ss.size))
            for i in range(x_ss.size):   

                x0[:,i] = x_ss.ravel().copy()
                x0[i,i] += h

            errors = self._path_obj(x0,parallel=True,use_jac_hh=True)

            jac[:,:] = (errors.reshape((x_ss.size,x_ss.size))-base)/h

        else:

            for i in range(x_ss.size):   

                x0 = x_ss.ravel().copy()
                x0[i] += h
                
                jac[:,i] = (self._path_obj(x0,use_jac_hh=True)-base)/h
        
        if do_print: print(f'full Jacobian computed in {elapsed(t0)}')

        # reset
        self.path = path_original

    ##########################
    # 5. find transiton path #
    ##########################   

    def _set_inputs_exo(self):
        """ set exogenous inputs based on shock specification """
        
        for inputname in self.inputs_exo:

            # a. jump and rho
            jumpname = f'jump_{inputname}'
            rhoname = f'rho_{inputname}'

            jump = getattr(self.par,jumpname)
            rho = getattr(self.par,rhoname)

            # b. factor
            T = self.par.transition_T//2
            fac = np.exp(jump*rho**np.arange(T))

            # c. apply
            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            
            patharray[:,:T] = ssvalue*fac[np.newaxis,:]
            patharray[:,T:] = ssvalue

    def evaluate_path(self,threads=1,use_jac_hh=False):
        """ evaluate transition path """

        with jit(self) as model:
            transition_path.evaluate_path(
                model.par,model.sol,model.sim,
                model.ss,model.path,
                model.jac_hh,threads=threads,use_jac_hh=use_jac_hh)   

    def evaluate_hh_frac_path(self,lower,upper,threads=1):
        """ evaluate transition path """
        
        D_copy = deepcopy(self.sim.D)
        sol_path_i_copy = deepcopy(self.sol.path_i)
        sol_path_w_copy = deepcopy(self.sol.path_w) 

        with jit(self) as model:
            transition_path.evaluate_hh_frac_path(
                model.par,model.sol,model.sim,model.path,
                lower=lower,upper=upper,threads=threads)  

        self.sim.D[:] = D_copy[:]
        self.sol.path_i[:] = sol_path_i_copy[:]
        self.sol.path_w[:] = sol_path_w_copy[:]

    def _path_obj(self,x,use_jac_hh=False,parallel=False,do_print=False):
        """ objective when solving for transition path """
        
        if parallel: 
            assert use_jac_hh
            assert not x is None

        par = self.par
        path = self.path

        if parallel:

            # a. set path for endogenous inputs
            x = x.reshape((len(self.inputs_endo),par.transition_T,len(self.inputs_endo)*self.par.transition_T))
            for i,varname in enumerate(self.inputs_endo):
                array = getattr(path,varname)                    
                array[:,:] = x[i,:,:].T

            # b. evaluate
            self.evaluate_path(threads=len(self.inputs_endo)*self.par.transition_T,use_jac_hh=use_jac_hh)

            # c. errors
            errors = np.zeros((len(self.targets),self.par.transition_T,len(self.inputs_endo)*self.par.transition_T))
            for i,varname in enumerate(self.targets):
                errors[i,:,:] = getattr(self.path,varname).T

            return errors

        else:

            # a. set path for endogenous inputs
            if not x is None:
                x = x.reshape((len(self.inputs_endo),par.transition_T))
                for i,varname in enumerate(self.inputs_endo):
                    array = getattr(path,varname)                    
                    array[0,:] = x[i,:]

            # b. evaluate
            self.evaluate_path(use_jac_hh=use_jac_hh)

            # c. errors
            errors = np.zeros((len(self.targets),self.par.transition_T))
            for i,varname in enumerate(self.targets):
                errors[i,:] = getattr(self.path,varname)[0,:]

            if do_print: 
                
                max_abs_error = np.max(np.abs(errors))

                for k in self.targets:
                    v = getattr(self.path,k)
                    print(f'{k:10s} = {np.max(np.abs(v)):12.8f}')

                print(f'\nmax abs. error: {max_abs_error:12.8f}')
        
            return errors.ravel()

    def find_transition_path(self,do_print=False):
        """ find transiton path """

        par = self.par

        t0 = time.time()

        # a. set path for exogenous inputs
        self._set_inputs_exo()

        # b. set initial value of endogenous inputs to ss
        x0 = np.zeros((len(self.inputs_endo),par.transition_T))
        for i,varname in enumerate(self.inputs_endo):
            x0[i,:] = getattr(self.ss,varname)

        # c. solve
        f = lambda x: self._path_obj(x)

        if do_print: print(f'finding the transition path:')
        x = broyden_solver(f,x0,self.jac,tol=par.tol_broyden,max_iter=par.max_iter_broyden,do_print=do_print)
        
        # d. final evaluation
        if do_print: print(f'')
        _errors = self._path_obj(x,do_print=do_print)

        if do_print: print(f'\ntransition path found in {elapsed(t0)}')

    ########
    # IRFs #
    ########

    def show_IRFs(self,paths,abs_value=None,lvl_value=None,facs=None,do_inputs=True,do_targets=True,T_max=None,ncols=4,filename=None):
        """ shows IRFS """

        # paths: list[str], variable names
        # abs_value: list[str], variable names to be shown as absolute difference to ss (defaulte is in % if ss is not nan)
        # lvl_value: list[str], variable names to be shown level (defaulte is in % if ss is not nan)
        # facs: dict[str -> float], scaling factor when in abs_value or lvl_value
        # do_inputs: boolean, show IRFs for the inputs
        # do_targets: boolean, show IRFs for the targets
        # T_max: int, length of IRF
        # ncols: number of columns
        # filename: filename if saving figure

        models = [self]
        labels = [None]
        show_IRFs(models,labels,paths,abs_value=abs_value,lvl_value=lvl_value,facs=facs,do_inputs=do_inputs,do_targets=do_targets,T_max=T_max,ncols=ncols,filename=filename)

    def compare_IRFs(self,models,labels,paths,abs_value=None,lvl_value=None,facs=None,do_inputs=True,do_targets=True,T_max=None,ncols=4,filename=None):
        """ compare IRFs across models """

        # models: list[GEModelClass], models
        # labels: list[str], model names
        # paths: list[str], variable names
        # abs_value: list[str], variable names to be shown as absolute difference to ss (defaulte is in % if ss is not nan)
        # lvl_value: list[str], variable names to be shown level (defaulte is in % if ss is not nan)
        # facs: dict[str -> float], scaling factor when in abs_value or lvl_value
        # do_inputs: boolean, show IRFs for the inputs
        # do_targets: boolean, show IRFs for the targets
        # T_max: int, length of IRF
        # ncols: number of columns
        # filename: filename if saving figure
                 
        show_IRFs(models,labels,paths,abs_value=abs_value,lvl_value=lvl_value,facs=facs,do_inputs=do_inputs,do_targets=do_targets,T_max=T_max,ncols=ncols,filename=filename)
