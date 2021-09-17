# contains main GEModelClass

import time
from types import SimpleNamespace
from copy import deepcopy
import numpy as np

from consav import jit
from consav.misc import elapsed

from .simulate import find_i_and_w_1d_1d, find_i_and_w_1d_1d_path
from .simulate import simulate_hh_forwards, simulate_hh_forwards_transpose
from .simulate import simulate_hh_path, simulate_hh_ss
from .broyden_solver import broyden_solver
from .figures import show_IRFs
from .determinancy import winding_criterion

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
            
            household_problem.solve_hh_path(model.par,model.sol,model.path)

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

    ################
    # 4. Jacobians #
    ################

    def _set_inputs_hh_ss(self):
        """ set household inputs to steady state """

        for inputname in self.inputs_hh:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

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

    def _calc_jac_hh_simple(self,shockname,dshock=1e-4,do_print=False,jac_hh=None):
        """ compute jacobian of household problem """

        par = self.par
        sol = self.sol
        sim = self.sim
        jac_hh = self.jac_hh if jac_hh is None else jac_hh
        
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
        sol_shock = deepcopy(self.sol)

        # c. simulate
        for s in range(par.transition_T):

            if do_print and s%100 == 0: print(f' {s}',end='')
            
            # i. before shock only time to shock matters
            sol.path_i[:s+1] = sol_shock.path_i[par.transition_T-(s+1):]
            sol.path_w[:s+1] = sol_shock.path_w[par.transition_T-(s+1):]

            for outputname in self.outputs_hh:
                varname = f'path_{outputname}'                     
                sol.__dict__[varname][:s+1] = sol_shock.__dict__[varname][par.transition_T-(s+1):]

            # ii. after shock solution is ss
            sol.path_i[s+1:] = sol.i
            sol.path_w[s+1:] = sol.w

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

                    if t == 0: # steady state
                        D_lag = sim.D
                    else:
                        D_lag = sim.path_D[t-1]

                    basevalue = np.sum(sol.__dict__[outputname]*sim.D)
                    shockvalue = np.sum(sol.__dict__[varname][t]*D_lag)

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
        for s in range(par.transition_T):
            
            t_ = (par.transition_T-1) - s
            simulate_hh_forwards(sim.D,sol.path_i[t_],sol.path_w[t_],par.z_trans_path[t_].T,diffs['D'][s])
            
            diffs['D'][s] = (diffs['D'][s]-sim.D)/dshock

            for outputname in self.outputs_hh:

                varname = f'path_{outputname}'

                basevalue = np.sum(sol.__dict__[outputname]*sim.D)
                shockvalue = np.sum(sol.__dict__[varname][t_]*sim.D)
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
        
    def compute_jac_hh(self,dshock=1e-4,do_print=False,do_print_full=False,do_simple=False):
        """ compute jacobian of household problem """

        path_original = deepcopy(self.path)

        # a. ghost run
        jac_hh_ghost = SimpleNamespace()
        if do_simple:
            self._calc_jac_hh_simple('__ghost',dshock=dshock,do_print=do_print,jac_hh=jac_hh_ghost)
        else:
            self._calc_jac_hh_fakenews('__ghost',dshock=dshock,do_print=do_print,do_print_full=do_print_full,jac_hh=jac_hh_ghost)

        # b. run for each input
        for shockname in self.inputs_hh:
            if do_simple:
                self._calc_jac_hh_simple(shockname,dshock=dshock,do_print=do_print)
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

                x0[i,:] = x_ss.ravel().copy()
                x0[i,i] += h

            errors = self._path_obj(x0,parallel=True,use_jac_hh=True)

            jac[:,:] = (errors.reshape((x_ss.size,x_ss.size)).T-base)/h

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
                array[:,:] = x[i,:,:]

            # b. evaluate
            self.evaluate_path(threads=len(self.inputs_endo)*self.par.transition_T,use_jac_hh=use_jac_hh)

            # c. errors
            errors = np.zeros((len(self.targets),self.par.transition_T,len(self.inputs_endo)*self.par.transition_T))
            for i,varname in enumerate(self.targets):
                errors[i,:,:] = getattr(self.path,varname)

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

    def show_IRFs(self,paths,abs_value=[],do_inputs=True,do_targets=True,T_max=None):
        """ shows IRFS """

        # paths: list[str], variable names
        # abs_value: list[str], variable names to be shown as absolute difference to ss (defaulte is in % if ss is not nan)
        # do_inputs: boolean, show IRFs for the inputs
        # do_targets: boolean, show IRFs for the targets
        # T_max: int, length of IRF

        models = [self]
        labels = [None]
        show_IRFs(models,labels,paths,abs_value=abs_value,do_inputs=do_inputs,do_targets=do_targets,T_max=T_max)

    def compare_IRFs(self,models,labels,paths,abs_value=[],do_inputs=True,do_targets=True,T_max=None):
        """ compare IRFs across models """

        # models: list[GEModelClass], models
        # labels: list[str], model names
        # paths: list[str], variable names
        # abs_value: list[str], variable names to be shown as absolute difference to ss (defaulte is in % if ss is not nan)
        # do_inputs: boolean, show IRFs for the inputs
        # do_targets: boolean, show IRFs for the targets
        # T_max: int, length of IRF
                 
        show_IRFs(models,labels,paths,abs_value=abs_value,do_inputs=do_inputs,do_targets=do_targets,T_max=T_max)

    ################
    # determinancy #
    ################

    def check_determinancy(self):
        """ check determinancy """

        # a. re-arrange Jacobian from var1 at time t=0,..., then var2 at time t=0,... to var1 at time t=1, var2 at time t=1,... (correct?)
        
        Nvars = len(self.inputs_endo)
        T = self.jac.shape[0]//Nvars
        temp = np.zeros(self.jac.shape)
        for var in range(Nvars):
            for t in range(T):
                temp[t*Nvars+var, T*var+t] = 1.0
        jac_new = temp@self.jac@np.linalg.inv(temp)  

        # b. pick out the A matrices (in practice finitely many)
        leadlags = 100 # Magic number. Truncation assumption that 100 periods away don't matter
        t_center = T-leadlags - 1

        A = np.zeros((2*leadlags+1,Nvars,Nvars))
        for i,j in enumerate(range(-leadlags, leadlags+1)):
            A[i,:,:] = jac_new[(Nvars*t_center):(Nvars*(t_center+1)), (Nvars*(t_center+j)):(Nvars*(t_center+j+1))]

        # c. call winding number function from Auclert et al. (2021)
        winding_number = winding_criterion(A, N=4096)

        print(f'{winding_number = }')

        # d. alternative way of calculating winding number
        
        js = np.arange(-leadlags,leadlags+1) 

        def calA(lamda, A, js):
            return np.linalg.det(np.sum([A[k]*np.exp(1j*js[k]*lamda) for k in range(js.size)], axis=0))

        gridpoints = 10**3
        output = [calA(t, A, js) for t in np.linspace(0, 2*np.pi, gridpoints)]
        output_real = [x.real for x in output]
        output_imag = [x.imag for x in output]
        output_real_pos = np.array(output_real)>=0
        output_imag_pos = np.array(output_imag)>=0
        output_imag_neg = np.array(output_imag)<=0
        
        up_passing = np.sum(output_real_pos[:-1]*output_imag_pos[1:]*output_imag_neg[:-1])
        down_passing = np.sum(output_real_pos[:-1]*output_imag_pos[:-1]*output_imag_neg[1:])

        print(f'{up_passing = }')
        print(f'{down_passing = }')
