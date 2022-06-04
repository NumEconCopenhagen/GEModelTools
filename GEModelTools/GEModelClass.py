# contains main GEModelClass

import time
from copy import deepcopy
import numpy as np

from EconModel import jit
from consav.misc import elapsed

from . import tests
from .simulate_hh import find_i_and_w_1d_1d, find_i_and_w_1d_1d_path
from .simulate_hh import simulate_hh_forwards_endo, simulate_hh_forwards_exo
from .simulate_hh import simulate_hh_forwards_endo_transpose, simulate_hh_forwards_exo_transpose
from .simulate_hh import simulate_hh_ss, simulate_hh_path, simulate_hh_z_path
from .broyden_solver import broyden_solver
from .simulate import update_IRF_hh,simulate_agg,simulate_agg_hh
from .figures import show_IRFs

class GEModelClass:

    ############
    # 1. setup #
    ############

    def allocate_GE(self,update_hh=True,ss_nan=True):
        """ allocate GE variables """

        # a1. input checks
        ns_attrs = ['par','ini','ss','path','sim']
        for ns in ns_attrs:
            assert ns in self.namespaces, f'{ns} must be a namespace'

        par = self.par
        ini = self.ini
        ss = self.ss
        path = self.path
        sim = self.sim

        varlist_attrs = [
            'grids_hh','pols_hh','inputs_hh','inputs_hh_z','outputs_hh','intertemps_hh',
            'shocks','unknowns','targets','varlist'
        ]

        for varlist in varlist_attrs:

            assert hasattr(self,varlist), f'.{varlist} must be defined, list[str]'
            strlist = getattr(self,varlist)
            assert type(strlist) is list, f'.{varlist} must be a list, but is {type(strlist)}'
            for el in varlist:
                assert type(el) is str, f'.{varlist} must be a list of strings'

        for attr in ['solve_hh_backwards','block_pre','block_post']:
            assert hasattr(self,attr), f'.{attr} must be defined'

        if len(self.inputs_hh_z) > 0:
            for attr in ['fill_z_trans']:
                assert hasattr(self,attr),f'.{attr} must be defined (when inputs_hh_z is not empty)'            

        # ensure stuff is saved
        for attr in ns_attrs + varlist_attrs + ['jac','H_U','H_Z','jac_hh','IRF']:
            if not attr in self.other_attrs:
                self.other_attrs.append(attr)

        # all houehold inputs
        self.inputs_hh_all = sorted(list(set(self.inputs_hh+self.inputs_hh_z)))

        # a2. naming checks
        for varname in self.inputs_hh_all:
            assert varname in self.varlist, f'{varname} not in .varlist'

        for varname in self.grids_hh: 
            assert not varname in self.varlist, f'{varname} is both in .grids_hh and .varlist'
        
        for varname in self.pols_hh: 
            assert not varname in self.varlist, f'{varname} is both in .pols_hh and .varlist'
        
        for varname in self.outputs_hh: 
            assert not varname in self.varlist, f'{varname} is both in .outputs_hh and .varlist'
        
        for varname in self.intertemps_hh: 
            assert not varname in self.varlist, f'{varname} is both in .intertemps_hh and .varlist'

        for varname in self.pols_hh:
            assert varname in self.outputs_hh, f'{varname} is in .pols_hh, but not .outputs_hh'

        for varname in self.outputs_hh: 
            Varname_hh = f'{varname.upper()}_hh'
            assert Varname_hh in self.varlist, f'{Varname_hh} is not in .varlist, but {varname} is in .outputs_hh'
            
        for varname in self.shocks + self.unknowns + self.targets:
            assert varname in self.varlist, f'{varname} not in .varlist'

        restricted_varnames = ['z_trans','Dz','D','pol_indices','pol_weights']
        
        for varname in self.varlist:
            assert not varname in ['restricted_varnames'], f'{varname} is not allowed in .varlist'

        for varname in self.outputs_hh:
            assert not varname in ['restricted_varnames'], f'{varname} is not allowed in .outputs_hh'
            
        # a3. dimensional checks
        sol_shape_endo = []
        for varname in self.grids_hh:
            Nx = f'N{varname}'
            assert hasattr(par,Nx), f'{Nx} not in .par'            
            sol_shape_endo.append(par.__dict__[Nx])

        assert hasattr(self.par,'Nfix'), 'par.Nfix must be specified'
        assert hasattr(self.par,'Nz'), 'par.Nz must be specified'

        sol_shape = (par.Nfix,par.Nz,*sol_shape_endo)
                
        # b. defaults in par
        par.__dict__.setdefault('T',500)
        par.__dict__.setdefault('simT',1_000)
        par.__dict__.setdefault('max_iter_solve',50_000)
        par.__dict__.setdefault('max_iter_simulate',50_000)
        par.__dict__.setdefault('max_iter_broyden',100)
        par.__dict__.setdefault('tol_solve',1e-12)
        par.__dict__.setdefault('tol_simulate',1e-12)
        par.__dict__.setdefault('tol_broyden',1e-10)

        for varname in self.shocks:
            par.__dict__.setdefault(f'jump_{varname}',0.0)
            par.__dict__.setdefault(f'rho_{varname}',0.0)

        # c. allocate grids and transition matrices
        if update_hh:

            for varname in self.grids_hh:

                gridname = f'{varname}_grid' 
                Nx = par.__dict__[f'N{varname}']
                gridarray = np.zeros(Nx)
                setattr(par,gridname,gridarray)
                
            par.z_grid = np.zeros(par.Nz)

        # d. allocate household variables
        path_pol_shape = (par.T,*sol_shape)
        sim_pol_shape = (par.simT,*sol_shape)

        if update_hh:
            
            # i. ss and path
            for varname in self.outputs_hh + self.intertemps_hh:
                ss.__dict__[varname] = np.zeros(sol_shape)
                path.__dict__[varname] = np.zeros(path_pol_shape)

            ini.Dz = np.zeros((par.Nfix,par.Nz,))
            ini.Dbeg = np.zeros(sol_shape)

            ss.z_trans = np.zeros((par.Nfix,par.Nz,par.Nz))
            ss.Dz = np.zeros((par.Nfix,par.Nz,))
            ss.D = np.zeros(sol_shape)
            ss.Dbeg = np.zeros(sol_shape)
            ss.pol_indices = np.zeros(sol_shape,dtype=np.int_)
            ss.pol_weights = np.zeros(sol_shape)

            path.z_trans = np.zeros((par.T,par.Nfix,par.Nz,par.Nz))
            path.Dz = np.zeros((par.T,par.Nfix,par.Nz))
            path.D = np.zeros(path_pol_shape)
            path.Dbeg = np.zeros(path_pol_shape)
            path.pol_indices = np.zeros(path_pol_shape,dtype=np.int_)
            path.pol_weights = np.zeros(path_pol_shape)

            # ii. sim
            for polname in self.pols_hh:
                sim.__dict__[polname] = np.zeros(sim_pol_shape)
                sim.__dict__[f'{polname.upper()}_hh_from_D'] = np.zeros(par.simT)

            sim.D = np.zeros(sim_pol_shape)
            sim.Dbeg = np.zeros(sim_pol_shape)
            sim.pol_indices = np.zeros(sim_pol_shape,dtype=np.int_)
            sim.pol_weights = np.zeros(sim_pol_shape)

        # e. allocate path and sim variables
        path_shape = (max(len(self.unknowns),len(self.shocks))*par.T,par.T)
        for varname in self.varlist:
            if ss_nan: 
                ss.__dict__[varname] = np.nan
                ini.__dict__[varname] = np.nan
            path.__dict__[varname] = np.zeros(path_shape)
            sim.__dict__[f'd{varname}'] = np.zeros(par.simT)

        # f. allocate Jacobians
        if update_hh:
            self.jac_hh = {}
            for outputname in self.outputs_hh:
                for inputname in self.inputs_hh:
                    key = (f'{outputname.upper()}_hh',inputname)
                    self.jac_hh[key] = np.zeros((par.T,par.T))            

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

        # g. allocate IRFs
        self.IRF = {}
        for varname in self.varlist:
            self.IRF[varname] = np.repeat(np.nan,par.T)
            for shockname in self.shocks:
                self.IRF[(varname,shockname)] = np.repeat(np.nan,par.T)

    def prepare_hh_ss(self):
        """ prepare the household block to solve for steady state """

        raise NotImplementedError

    def print_unpack_varlist(self):
        """ print varlist for use in evaluate_path() """

        print(f'    for ncol in range(ncols):\n')
        print('        # unpack')
        for varname in self.varlist:
            print(f'        {varname} = path.{varname}[ncol,:]')

    def update_aggregate_settings(self,shocks=None,unknowns=None,targets=None):
        """ update aggregate settings and re-allocate jac etc. """
        
        if not shocks is None: self.shocks = shocks
        if not unknowns is None: self.unknowns = unknowns
        if not targets is None: self.targets = targets

        self.allocate_GE(update_hh=False,ss_nan=False)

    ####################
    # 2. steady state #
    ###################

    def _find_i_and_w_dict(self,ss_dict):
        """ find policy indices and weights from dict """

        par = self.par

        if len(self.grids_hh) == 1:
            pol1 = ss_dict[f'{self.grids_hh[0]}']
            grid1 = getattr(par,f'{self.grids_hh[0]}_grid')
            find_i_and_w_1d_1d(pol1,grid1,ss_dict['pol_indices'],ss_dict['pol_weights'])
        else:
            raise NotImplementedError

    def _find_i_and_w_ss(self):
        """ find policy indices and weights in steady state """

        self._find_i_and_w_dict(self.ss.__dict__)

    def _get_stepvars_hh_z_ss(self):
        """ get variables for backwards step in steady state """

        par = self.par
        ss = self.ss

        stepvars_hh = {'par':par,'z_trans':ss.z_trans}
        for varname in self.inputs_hh_z: stepvars_hh[varname] = getattr(ss,varname)

        return stepvars_hh

    def fill_z_trans_ss(self):
        """ fill transition matrix in steady state, ss.z_trans """

        with jit(self) as model:
            stepvars_hh_z_ss = self._get_stepvars_hh_z_ss()
            self.fill_z_trans(**stepvars_hh_z_ss)

    def _get_stepvars_hh_ss(self,outputs_inplace=True):
        """ get variables for backwards step in steady state """

        par = self.par
        ss = self.ss

        # a. inputs
        stepvars_hh = {'par':par,'z_trans':ss.z_trans}
        for varname in self.inputs_hh: stepvars_hh[varname] = getattr(ss,varname)
        for varname in self.intertemps_hh: stepvars_hh[f'{varname}_plus'] = getattr(ss,varname)

        # b. outputs
        for varname in self.outputs_hh + self.intertemps_hh: 
            if outputs_inplace:
                stepvars_hh[varname] = getattr(ss,varname)
            else:
                stepvars_hh[varname] = getattr(ss,varname).copy()

        return stepvars_hh

    def solve_hh_ss(self,do_print=False,initial_guess=None):
        """ solve the household problem in steady state """

        t0 = time.time()

        if initial_guess is None: initial_guess = {}

        # a. prepare model to find steady state
        self.prepare_hh_ss()

        # check
        for i_fix in range(self.par.Nfix):
            for i_z in range(self.par.Nz):
                rowsum = np.sum(self.ss.z_trans[i_fix,i_z,:])
                check = np.isclose(rowsum,1.0)
                assert check, f'sum(ss.z_trans[{i_fix},{i_z},:] = {rowsum:12.8f}, should be 1.0'

        # overwrite initial guesses
        for varname,value in initial_guess.items():
            self.ss.__dict__[varname][:] = value

        # b. solve backwards until convergence
        with jit(self,show_exc=False) as model:
            
            par = model.par
            ss = model.ss

            it = 0
            while True:

                # i. old policy
                old = {pol:getattr(ss,pol).copy() for pol in self.pols_hh}

                # ii. step backwards
                stepvars_hh = self._get_stepvars_hh_ss()
                self.solve_hh_backwards(**stepvars_hh)

                # iii. check change in policy
                max_abs_diff = max([np.max(np.abs(getattr(ss,pol)-old[pol])) for pol in self.pols_hh])
                if max_abs_diff < par.tol_solve: 
                    break
                
                # iv. increment
                it += 1
                if it > par.max_iter_solve: 
                    raise ValueError('solve_hh_ss(), too many iterations')

        if do_print: print(f'household problem in ss solved in {elapsed(t0)} [{it} iterations]')

        # c. indices and weights    
        self._find_i_and_w_ss()

    def simulate_hh_ss(self,do_print=False,Dbeg=None,find_i_and_w=False):
        """ simulate the household problem in steady state """
        
        par = self.par
        ss = self.ss

        t0 = time.time()

        if find_i_and_w: self._find_i_and_w_ss()

        # a. initial guess
        if not Dbeg is None:
            ss.Dbeg[:] = Dbeg

        # check
        Dbeg_sum = np.sum(ss.Dbeg)
        assert np.isclose(Dbeg_sum,1.0), f'sum(ss.Dbeg) = {Dbeg_sum:12.8f}, should be 1.0'
        
        # b. simulate
        with jit(self,show_exc=False) as model:

            par = model.par
            ss = model.ss
            
            it = simulate_hh_ss(par,ss)

        # c. Dz
        ss.Dz[:,:] = np.sum(ss.Dbeg,axis=tuple(range(2,ss.Dbeg.ndim)))

        if do_print: print(f'household problem in ss simulated in {elapsed(t0)} [{it} iterations]')

    def find_ss(self,do_print=False):
        """ solve for the steady state """

        raise NotImplementedError

    #####################
    # 3. household path #
    #####################

    def _find_i_and_w_path(self):
        """ find indices and weights along the transition path"""

        par = self.par
        path = self.path

        if len(self.grids_hh) == 1:
            path_pol1 = getattr(path,f'{self.grids_hh[0]}')
            grid1 = getattr(par,f'{self.grids_hh[0]}_grid') 
            find_i_and_w_1d_1d_path(par.T,path_pol1,grid1,path.pol_indices,path.pol_weights)
        else:
            raise NotImplemented

    def _get_stepvars_hh_z_path(self,t):
        """ get variables for backwards step in along transition path"""

        par = self.par
        path = self.path

        stepvars_hh_z = {'par':par,'z_trans':path.z_trans[t]}
        for varname in self.inputs_hh_z: stepvars_hh_z[varname] = getattr(path,varname)[0,t]

        return stepvars_hh_z

    def _get_stepvars_hh_path(self,t):
        """ get variables for backwards step in along transition path"""

        par = self.par
        path = self.path
        ss = self.ss

        stepvars_hh = {'par':par,'z_trans':path.z_trans[t]}
        for varname in self.inputs_hh: stepvars_hh[varname] = getattr(path,varname)[0,t]
        for varname in self.outputs_hh + self.intertemps_hh: stepvars_hh[varname] = getattr(path,f'{varname}')[t]
        
        for varname in self.intertemps_hh: 
            if t < par.T-1:
                value_plus = getattr(path,f'{varname}')[t+1]
            else:
                value_plus = getattr(ss,f'{varname}')

            stepvars_hh[f'{varname}_plus'] = value_plus

        return stepvars_hh

    def solve_hh_path(self,do_print=False):
        """ solve the household problem along the transition path """

        t0 = time.time()

        with jit(self,show_exc=False) as model:

            par = model.par
            ss = model.ss
            path = model.path
            
            for k in range(par.T):

                t = (par.T-1)-k

                # i. update transition matrix
                if len(self.inputs_hh_z) > 0:
                    stepvars_hh_z = self._get_stepvars_hh_z_path(t)
                    self.fill_z_trans(**stepvars_hh_z)
                else:
                    path.z_trans[t,:] = ss.z_trans

                # ii. stepbacks
                stepvars_hh = self._get_stepvars_hh_path(t)
                self.solve_hh_backwards(**stepvars_hh)

        # c. indices and weights
        self._find_i_and_w_path()

        if do_print: print(f'household problem solved along transition path in {elapsed(t0)}')

    def simulate_hh_path(self,do_print=False,find_i_and_w=False,Dbeg=None):
        """ simulate the household problem along the transition path"""
        
        t0 = time.time() 

        if find_i_and_w: self._find_i_and_w_path()

        # a. initial distribution
        if Dbeg is None:
            self.path.Dbeg[0] = self.ss.Dbeg
        else:
            self.path.Dbeg[0] = Dbeg

        # check
        Dbeg_sum = np.sum(self.path.Dbeg[0])
        assert np.isclose(Dbeg_sum,1.0), f'sum(path.Dbeg[0]) = {Dbeg_sum:12.8f}, should be 1.0'

        # b. simulate
        with jit(self,show_exc=False) as model:
            simulate_hh_path(model.par,model.path)

        # c. Dz
        self.path.Dz[:,:,:] = np.sum(self.path.Dbeg,axis=tuple(range(3,self.path.Dbeg.ndim)))

        if do_print: print(f'household problem simulated along transition in {elapsed(t0)}')

    def simulate_hh_z_path(self,do_print=False,update_z=True,Dz_ini=None):
        """ simulate z along the transition path"""
        
        t0 = time.time() 

        par = self.par
        ss = self.ss
        path = self.path

        # a. initial distribution
        if Dz_ini is None: Dz_ini = self.ss.Dz

        # check
        Dz_ini_sum = np.sum(Dz_ini)
        assert np.isclose(Dz_ini_sum,1.0), f'sum(Dz_ini[0]) = {Dz_ini_sum:12.8f}, should be 1.0'

        # b. simulate
        with jit(self,show_exc=False) as model:

            if update_z:

                for t in range(par.T):

                    if len(self.inputs_hh_z) > 0:
                        stepvars_hh_z = self._get_stepvars_hh_z_path(t)
                        self.fill_z_trans(**stepvars_hh_z)
                    else:
                        path.z_trans[t,:] = ss.z_trans

            simulate_hh_z_path(model.par,model.path,Dz_ini)

        if do_print: print(f'household z z simulated along transition in {elapsed(t0)}')

    def _set_inputs_hh_all_ss(self):
        """ set household inputs to steady state """

        for inputname in self.inputs_hh_all:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    def decompose_hh_path(self,do_print=False,Dbeg=None,use_inputs=None):
        """ decompose household transition path wrt. inputs or initial distribution """

        ss = self.ss

        if use_inputs is None: 
            use_inputs_list = []
        elif use_inputs == 'all':
            use_inputs_list = self.inputs_hh_all
        else:
            use_inputs_list = use_inputs

        # a. save original path and create clean path
        path_original = self.path
        path = self.path = deepcopy(self.path)

        for varname in self.varlist:
            if varname in self.inputs_hh_all: continue 
            path.__dict__[varname][:] = np.nan

        # b. set inputs
        for varname in self.inputs_hh_all:
            if varname in use_inputs_list: continue
            path.__dict__[varname][0,:] = ss.__dict__[varname]
                
        # c. solve and simulate
        if not use_inputs == 'all':
            self.solve_hh_path(do_print=do_print)
        self.simulate_hh_path(do_print=do_print,Dbeg=Dbeg)

        # d. aggregates
        for outputname in self.outputs_hh:

            Outputname_hh = f'{outputname.upper()}_hh'
            pathvalue = path.__dict__[Outputname_hh]

            pol = path.__dict__[outputname]
            pathvalue[0,:] = np.sum(pol*path.D,axis=tuple(range(1,pol.ndim)))
            
        return_path = self.path

        # e. restore original path
        self.path = path_original

        return return_path

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
        ss = self.ss
        path = self.path
        
        if s_list is None: s_list = list(range(par.T))

        t0 = time.time()
        if do_print: print(f'finding Jacobian wrt. {inputname:15s}:',end='')
            
        # a. allocate
        for outputname in self.outputs_hh:
            jac_hh[(f'{outputname.upper()}_hh',inputname)] = np.zeros((par.T,par.T))

        # b. solve with shock in last period
        self._set_inputs_hh_all_ss()

        if not inputname == 'ghost':
            shockarray = getattr(self.path,inputname)
            shockarray[0,-1] += dx

        self.solve_hh_path()

        # c. simulate
        path_shock = deepcopy(self.path)

        for s in s_list:

            if do_print: print(f' {s}',end='')
            
            # i. before shock only time to shock matters
            path.z_trans[:s+1] = path_shock.z_trans[par.T-(s+1):]
            path.pol_indices[:s+1] = path_shock.pol_indices[par.T-(s+1):]
            path.pol_weights[:s+1] = path_shock.pol_weights[par.T-(s+1):]

            for outputname in self.outputs_hh:
                path.__dict__[outputname][:s+1] = path_shock.__dict__[outputname][par.T-(s+1):]

            # ii. after shock solution is ss
            path.pol_indices[s+1:] = ss.pol_indices
            path.pol_weights[s+1:] = ss.pol_weights
            path.z_trans[s+1:] = ss.z_trans

            for outputname in self.outputs_hh:
                path.__dict__[outputname][s+1:] = ss.__dict__[outputname]

            # iii. simulate path
            self.simulate_hh_path()

            # iv. compute Jacobian
            for outputname in self.outputs_hh:
                
                jac_hh_ = jac_hh[(f'{outputname.upper()}_hh',inputname)]

                for t in range(par.T):

                    basevalue = np.sum(ss.__dict__[outputname]*ss.D)
                    shockvalue = np.sum(path.__dict__[outputname][t]*path.D[t])

                    jac_hh_[t,s] = (shockvalue-basevalue)/dx

        if do_print: print(f' [computed in {elapsed(t0)}]')

    def _calc_jac_hh_fakenews(self,inputs_hh_all=None,dx=1e-4,do_print=False):
        """ compute Jacobian of household problem with fake news algorithm """

        if inputs_hh_all is None: inputs_hh_all = self.inputs_hh_all

        with jit(self) as model:

            par = model.par
            ss = model.ss
            path = model.path

            t0_all = time.time()

            # step 1a: one step deviation from steady state
            t0 = time.time()

            # i. solve one step backwards
            one_step_ss = self._get_stepvars_hh_ss(outputs_inplace=False)
            self.solve_hh_backwards(**one_step_ss)

            # ii. outputs
            for outputname in self.outputs_hh:
                Outputname_hh = f'{outputname.upper()}_hh'
                one_step_ss[Outputname_hh] = np.sum(one_step_ss[outputname]*ss.D)

            # ii. simulate one step forwards
            one_step_ss['Dbeg_plus'] = np.zeros(ss.Dbeg.shape)
            one_step_ss['pol_indices'] = np.zeros(ss.pol_indices.shape,dtype=np.int_)
            one_step_ss['pol_weights'] = np.zeros(ss.pol_weights.shape)

            self._find_i_and_w_dict(one_step_ss)
            simulate_hh_forwards_endo(ss.D,one_step_ss['pol_indices'],one_step_ss['pol_weights'],one_step_ss['Dbeg_plus'])

            if do_print: print(f'one step deviation from steady state calculated in {elapsed(t0)}')

            # step 1b: full backwards iteration

            # allocate
            curly_Y0 = {}
            curly_Dbeg1 = {}

            # loop over inputs
            self.dpols = {}
            for inputname in inputs_hh_all:
                
                for outputname in self.outputs_hh:
                    Outputname_hh = f'{outputname.upper()}_hh'
                    curly_Y0[(Outputname_hh,inputname)] = np.zeros(par.T) 

                curly_Dbeg1[inputname] = np.zeros(path.Dbeg.shape)

                t0 = time.time()

                # backwards iteration
                dintertemps = {}

                for polname in self.pols_hh:
                    self.dpols[(polname,inputname)] = np.zeros(getattr(path,polname).shape)

                for s in range(par.T):
                    
                    # i. solve gradually backwards
                    stepvars_hh = self._get_stepvars_hh_ss(outputs_inplace=False)

                    do_z_trans = (s == 0) and (len(self.inputs_hh_z) > 0) and (inputname in self.inputs_hh_z)
                    
                    if s == 0:
                        
                        if inputname in self.inputs_hh:
                            stepvars_hh[inputname] += dx

                        if do_z_trans:

                            # o. fill
                            stepvars_hh_z = self._get_stepvars_hh_z_ss()
                            stepvars_hh_z[inputname] += dx
                            stepvars_hh_z['z_trans'] = np.zeros(ss.z_trans.shape)
                            self.fill_z_trans(**stepvars_hh_z)
                             
                            # oo. transfer
                            stepvars_hh['z_trans'] = stepvars_hh_z['z_trans']

                    else:

                        for varname in self.intertemps_hh:
                            varname_plus = f'{varname}_plus'
                            stepvars_hh[varname_plus] = stepvars_hh[varname_plus] + dintertemps[varname]

                    self.solve_hh_backwards(**stepvars_hh)

                    for varname in self.intertemps_hh:
                        dintertemps[varname] = stepvars_hh[varname]-one_step_ss[varname]

                    for polname in self.pols_hh:
                        self.dpols[(polname,inputname)][s] = (stepvars_hh[polname]-one_step_ss[polname])/dx

                    if do_z_trans:

                        D0 = np.zeros(ss.D.shape)
                        z_trans_T = np.transpose(stepvars_hh_z['z_trans'],axes=(0,2,1)).copy()
                        simulate_hh_forwards_exo(ss.Dbeg,z_trans_T,D0)

                    # ii. curly_Y0
                    for outputname in self.outputs_hh:
                        
                        Outputvalue_hh = np.sum(stepvars_hh[outputname]*ss.D)
                        Outputname_hh = f'{outputname.upper()}_hh'
                        curly_Y0[(Outputname_hh,inputname)][s] = (Outputvalue_hh-one_step_ss[Outputname_hh])/dx

                        if do_z_trans:

                            Outputvalue_hh = np.sum(getattr(ss,outputname)*D0)
                            add = (Outputvalue_hh-one_step_ss[Outputname_hh])/dx
                            curly_Y0[(Outputname_hh,inputname)][s] += add

                    # ii. curly_Dbeg1
                    stepvars_hh['Dbeg_plus'] = np.zeros(ss.Dbeg.shape)
                    stepvars_hh['pol_indices'] = np.zeros(ss.pol_indices.shape,dtype=np.int_)
                    stepvars_hh['pol_weights'] = np.zeros(ss.pol_weights.shape)

                    self._find_i_and_w_dict(stepvars_hh)
                    simulate_hh_forwards_endo(ss.D,stepvars_hh['pol_indices'],stepvars_hh['pol_weights'],stepvars_hh['Dbeg_plus'])

                    curly_Dbeg1[inputname][s] = (stepvars_hh['Dbeg_plus']-one_step_ss['Dbeg_plus'])/dx

                    if do_z_trans:

                        simulate_hh_forwards_endo(D0,ss.pol_indices,ss.pol_weights,stepvars_hh['Dbeg_plus'])
                        add = (stepvars_hh['Dbeg_plus']-one_step_ss['Dbeg_plus'])/dx
                        curly_Dbeg1[inputname][s] += add
 
                if do_print: print(f'curly_Y and curly_D calculated for {inputname:15s} in {elapsed(t0)}')

        # c. step 3: expectation factors
        t0 = time.time()
        
        # demeaning improves numerical stability
        def demean(x):
            return x - x.sum()/x.size

        curly_E = {}

        for outputname in self.outputs_hh:

            sol_ss = getattr(ss,outputname)
            
            curly_E[outputname] = np.zeros((par.T-1,*sol_ss.shape))
            temp = demean(sol_ss)
            simulate_hh_forwards_exo_transpose(temp,ss.z_trans,curly_E[outputname][0])

        for t in range(1,par.T-1):
            
            for outputname in self.outputs_hh:
                simulate_hh_forwards_endo_transpose(curly_E[outputname][t-1],ss.pol_indices,ss.pol_weights,temp)
                simulate_hh_forwards_exo_transpose(temp,ss.z_trans,curly_E[outputname][t])
                curly_E[outputname][t] = demean(curly_E[outputname][t])
            
        if do_print: print(f'curly_E calculated in {elapsed(t0)}')
            
        # d. step 4: F   
        t0 = time.time()

        for inputname in self.inputs_hh_all:
            for outputname in self.outputs_hh:

                Outputname_hh = f'{outputname.upper()}_hh'

                if inputname not in inputs_hh_all:

                    self.jac_hh[(f'{outputname.upper()}_hh',inputname)] = np.zeros((par.T,par.T))

                else:

                    F = np.zeros((par.T,par.T))

                    F[0,:] = curly_Y0[(Outputname_hh,inputname)]
                    F[1:,:] = curly_E[outputname].reshape((par.T-1,-1))@curly_Dbeg1[inputname].reshape((par.T,-1)).T

                    J = F
                    for t in range(1,J.shape[1]): J[1:,t] += J[:-1,t-1]

                    self.jac_hh[(f'{outputname.upper()}_hh',inputname)] = J

        if do_print: print(f'builiding blocks combined in {elapsed(t0)}')
        if do_print: print(f'household Jacobian computed in {elapsed(t0_all)}')
        
    def _compute_jac_hh(self,dx=1e-4,inputs_hh_all=None,do_print=False,do_direct=False,s_list=None):
        """ compute Jacobian of household problem """

        t0 = time.time()

        if inputs_hh_all is None: inputs_hh_all = self.inputs_hh_all
        if not do_direct: assert s_list is None, 'not implemented for fake news algorithm'

        self._path_fakenews = {}

        # a. fake news
        if not do_direct:
            
            self._calc_jac_hh_fakenews(dx=dx,inputs_hh_all=inputs_hh_all,do_print=do_print)

        # b. direct
        else:

            jac_hh = {}
            jac_hh_ghost = {}
            
            path_original = self.path
            self.path = deepcopy(self.path)

            # i. ghost
            self._calc_jac_hh_direct(jac_hh_ghost,'ghost',dx=dx,
                do_print=do_print,s_list=s_list)

            # ii. each input
            for inputname in inputs_hh_all:
                self._calc_jac_hh_direct(jac_hh,inputname,dx=dx,
                    do_print=do_print,s_list=s_list)
                
            # ii. correction with ghost run
            for outputname in self.outputs_hh:
                for inputname in self.inputs_hh_all:

                    key = (f'{outputname.upper()}_hh',inputname)
                    key_ghost = (f'{outputname.upper()}_hh','ghost')

                    if not inputname in inputs_hh_all: 
                        
                        self.jac_hh[key] = np.zeros((self.par.T,self.par.T))

                    else:    
                    
                        self.jac_hh[key] = jac_hh[key]-jac_hh_ghost[key_ghost]

            if do_print: print(f'household Jacobian computed in {elapsed(t0)}')

            # reset
            self.path = path_original

    def _compute_jac(self,inputs=None,dx=1e-4,do_print=False,parallel=True):
        """ compute full Jacobian """
        
        if type(inputs) is str:

            assert inputs in ['shocks','unknowns'], 'unknown string value of inputs'

            do_unknowns = inputs == 'unknowns'
            do_shocks = inputs == 'shocks'

        else:

            assert type(inputs) is list, 'inputs must be string or list'
            for el in list:
                assert type(el) is str, f'all elements in inputs must be string, {el} is {type(el)}'

            do_unknowns = False
            do_shocks = False

            if not parallel: raise NotImplementedError('must have parallel=True')

        t0 = time.time()
        evaluate_t = 0.0

        path_original = self.path
        self.path = deepcopy(self.path)

        if do_unknowns:
            inputs = self.unknowns
        elif do_shocks:
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
        elif do_shocks:
            jac_mat = self.H_Z
        else:
            jac_dict = {}
        
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
            self.evaluate_path(ncols=len(inputs)*par.T,use_jac_hh=True)
            evaluate_t += time.time()-t0_
            errors = self._get_errors(inputs,parallel=True)

            # iii. Jacobian
            if do_unknowns or do_shocks:
                jac_mat[:,:] = (errors.reshape(jac_mat.shape)-base[:,np.newaxis])/dx

            # iv. all other variables
            for i_input,inputname in enumerate(inputs):
                for outputname in self.varlist:
                    
                    key = (outputname,inputname)
                    if do_unknowns or do_shocks:
                        jac = self.jac[key]
                    else:
                        jac = jac_dict[key] = np.zeros((par.T,par.T))
                        
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
                jac_mat[:,i] = (errors.ravel()-base)/dx
        
        if do_print:
            if do_unknowns:
                print(f'full Jacobian to unknowns computed in {elapsed(t0)} [in evaluate_path(): {elapsed(0,evaluate_t)}]')
            else: 
                print(f'full Jacobian to shocks computed in {elapsed(t0)} [in evaluate_path(): {elapsed(0,evaluate_t)}]')

        # reset
        self.path = path_original

        if not (do_unknowns or do_shocks):
            return jac_dict
            
    def compute_jacs(self,dx=1e-4,skip_hh=False,inputs_hh_all=None,skip_shocks=False,do_print=False,parallel=True,do_direct=False):
        """ compute all Jacobians """
        
        if not skip_hh and len(self.outputs_hh) > 0:
            if do_print: print('household Jacobians:')
            self._compute_jac_hh(inputs_hh_all=inputs_hh_all,dx=dx,do_direct=do_direct,do_print=do_print)
            if do_print: print('')

        if do_print: print('full Jacobians:')
        self._compute_jac(inputs='unknowns',dx=dx,parallel=parallel,do_print=do_print)
        if not skip_shocks: self._compute_jac(inputs='shocks',dx=dx,parallel=parallel,do_print=do_print)

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
                patharray[:,:] = ssvalue 
                Tshock = self.par.T//2
                patharray[:,:Tshock] += scale*rho**np.arange(Tshock)

    def _set_ini(self,ini_input=None):
        """ set initial distribution """

        par = self.par
        ini = self.ini
        ss = self.ss

        # a. transfer information to ini
        if ini_input is None:

            pass

        elif ini_input == 'ss':
            
            for varname in self.varlist:
                ini.__dict__[varname] = ss.__dict__[varname]
            
            ini.Dbeg[:] = ss.Dbeg

        elif type(ini_input) is dict:

            for varname in self.varlist:
                
                if varname in ini_input:
                    ini.__dict__[varname] = ini_input[varname]
                else:
                    ini.__dict__[varname] = np.nan

            ini.Dbeg[:] = ini_input['Dbeg']

        else:
            
            raise NotImplemented('ini must be a dictionary')
            
        # b. derive Dz
        ini.Dz[:] = np.sum(ini.Dbeg,axis=tuple(range(2,ini.Dbeg.ndim)))

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

    def evaluate_path(self,ini='ss',ncols=1,use_jac_hh=False):
        """ evaluate transition path """

        par = self.par
        ss = self.ss
        path = self.path

        assert use_jac_hh or ncols == 1
        
        # a. update initial distribution
        self._set_ini(ini_input=ini)

        # b. before household block
        with jit(self,show_exc=False) as model:
            self.block_pre(model.par,model.ini,model.ss,model.path,ncols=ncols)

        # c. household block
        if use_jac_hh and len(self.outputs_hh) > 0: # linearized

            for outputname in self.outputs_hh:
                
                Outputname_hh = f'{outputname.upper()}_hh'

                # i. set steady state value
                pathvalue = path.__dict__[Outputname_hh]
                ssvalue = ss.__dict__[Outputname_hh]
                pathvalue[:,:] = ssvalue

                # ii. update with Jacobians and inputs
                for inputname in self.inputs_hh_all:

                    jac_hh = self.jac_hh[(f'{Outputname_hh}',inputname)]

                    ssvalue_input = ss.__dict__[inputname]
                    pathvalue_input = path.__dict__[inputname]

                    pathvalue[:,:] += (jac_hh@(pathvalue_input.T-ssvalue_input)).T 
                    # transposing needed for correct broadcasting
        
        elif len(self.outputs_hh) > 0: # non-linear solution
            
            # i. solve
            self.solve_hh_path()

            # ii. simulate
            self.simulate_hh_path(Dbeg=self.ini.Dbeg)

            # iii. aggregate
            for outputname in self.outputs_hh:

                Outputname_hh = f'{outputname.upper()}_hh'
                pathvalue = path.__dict__[Outputname_hh]

                pol = path.__dict__[outputname]
                pathvalue[0,:] = np.sum(pol*path.D,axis=tuple(range(1,pol.ndim)))

        else:

            pass # no household block
                
        # d. after household block
        with jit(self,show_exc=False) as model:
            self.block_post(model.par,model.ini,model.ss,model.path,ncols=ncols)

    def _evaluate_H(self,x,do_print=False):
        """ compute error in equation system for targets """
        
        # a. evaluate
        self._set_unknowns(x,self.unknowns)
        self.evaluate_path(ini=None)
        errors = self._get_errors() 
        
        # b. print
        if do_print: 
            
            max_abs_error = np.max(np.abs(errors))

            for k in self.targets:
                v = getattr(self.path,k)
                print(f'{k:15s} = {np.max(np.abs(v)):8.1e}')

            print(f'\nmax abs. error: {max_abs_error:8.1e}')

        # c. return as vector
        return errors.ravel()

    def find_transition_path(self,ini='ss',unknowns_ss=True,shock_specs=None,do_end_check=True,do_print=False,do_print_unknowns=False):
        """ find transiton path (fully non-linear) """

        par = self.par
        ss = self.ss
        path = self.path

        t0 = time.time()

        # a. set path for shocks
        self._set_shocks(shock_specs=shock_specs)

        # b. set initial value of unknows to ss
        x0 = np.zeros((len(self.unknowns),par.T))
        for i,varname in enumerate(self.unknowns):
            if unknowns_ss:
                x0[i,:] = getattr(self.ss,varname)
            else:
                x0[i,:] = getattr(self.path,varname)[0,:]

        # c. set initial state
        self._set_ini(ini_input=ini)

        # d. solve
        obj = lambda x: self._evaluate_H(x)

        if do_print: print(f'finding the transition path:')
        x = broyden_solver(obj,x0,self.H_U,
            tol=par.tol_broyden,
            max_iter=par.max_iter_broyden,
            model=self,
            do_print=do_print,
            do_print_unknowns=do_print_unknowns)
        
        # e. final evaluation
        self._evaluate_H(x)

        # f. test
        if do_end_check:
            for varname in self.varlist:
                ssval = ss.__dict__[varname]
                if np.isnan(ssval): continue
                endpathval = path.__dict__[varname][0,-1]
                if not np.isclose(ssval,endpathval):
                    print(f'{varname}: terminal value is {endpathval:12.8f}, but ss value is {ssval:12.8f}')

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

    def prepare_simulate(self,skip_hh=False,reuse_G_U=False,do_print=True):
        """ prepare model for simulation by calculating IRFs """

        par = self.par
        ss = self.ss
        path = self.path

        path_original = self.path
        path = self.path = deepcopy(self.path)

        t0 = time.time()

        # a. solution matrix
        t0_sol = time.time()
        if not reuse_G_U: self.G_U[:,:] = -np.linalg.solve(self.H_U,self.H_Z)       
        t1_sol = time.time()   
        
        # b. calculate unit shocks
        self._set_shocks(std_shock=True)

        # c. IRFs
        for i_input,shockname in enumerate(self.shocks):
            
            # i. shocks
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

                self.IRF[(varname,shockname)][:] = self.jac[(varname,shockname)]@self.IRF[(shockname,shockname)]
                for inputname in self.unknowns:
                    self.IRF[(varname,shockname)][:] += self.jac[(varname,inputname)]@self.IRF[(inputname,shockname)]

        # d. household
        if not skip_hh:

            t0_hh = time.time()

            self.IRF['pols'] = {}
            for shockname in self.shocks:  
                for polname in self.pols_hh:
                    IRF_pols = self.IRF['pols'][(polname,shockname)] = np.zeros((*ss.pol_indices.shape,par.T))
                    for inputname in self.inputs_hh_all:    
                        update_IRF_hh(IRF_pols,self.dpols[(polname,inputname)],self.IRF[(inputname,shockname)])
                        
            t1_hh = time.time()

        if do_print: 
            print(f'simulation prepared in {elapsed(t0)} [solution matrix: {elapsed(t0_sol,t1_sol)}',end='')
            if skip_hh:
                print(f']') 
            else:
                print(f', households: {elapsed(t0_hh,t1_hh)}]')

        # reset
        self.path = path_original

    def simulate(self,do_prepare=True,skip_hh=False,reuse_G_U=False,do_print=False):
        """ simulate the model """

        par = self.par
        ss = self.ss
        sim = self.sim
        
        # a. prepare simulation
        if do_prepare: self.prepare_simulate(skip_hh=skip_hh,reuse_G_U=reuse_G_U,do_print=do_print)

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
        if not skip_hh:

            t0 = time.time()

            # i. policies
            IRF_pols_mat = np.zeros((len(self.pols_hh),len(self.shocks),*ss.pol_indices.shape,par.T))
            for i,polname in enumerate(self.pols_hh):
                for j,shockname in enumerate(self.shocks):
                    IRF_pols_mat[i,j] = self.IRF['pols'][(polname,shockname)]

            sim_pols_mat = np.zeros((len(self.pols_hh),par.simT,*ss.pol_indices.shape))
            
            t0_ = time.time()
            simulate_agg_hh(epsilons,IRF_pols_mat,sim_pols_mat)
            t1_ = time.time()

            for i,polname in enumerate(self.pols_hh):
                sim_pols_mat[i] += ss.__dict__[polname]
                sim.__dict__[polname] = sim_pols_mat[i]

            if do_print: print(f'household policies simulated in {elapsed(t0)}')

            # ii. distribution
            self.simulate_distribution(do_print=do_print)

    def simulate_distribution(self,do_print=False):
        """ simulate distribution """

        par = self.par
        ss = self.ss
        sim = self.sim

        t0 = time.time()

        # a. initialize
        sim.Dbeg[0] = ss.Dbeg
        
        # b. transition matrix
        if len(self.inputs_hh_z) > 0:
            raise NotImplementedError
        else:
            z_trans_T = np.transpose(ss.z_trans,axes=(0,2,1)).copy()

        # c. simulate
        if len(self.grids_hh) == 1:

            grid1 = getattr(par,f'{self.grids_hh[0]}_grid')

            for t in range(par.simT):
            
                simulate_hh_forwards_exo(sim.Dbeg[t],z_trans_T,sim.D[t])    
                
                if t < par.simT-1:
                    sim_i = sim.pol_indices[t]
                    sim_w = sim.pol_weights[t]
                    sim_pol = sim.__dict__[self.pols_hh[0]]
                    find_i_and_w_1d_1d(sim_pol[t],grid1,sim_i,sim_w)
                    simulate_hh_forwards_endo(sim.D[t],sim_i,sim_w,sim.Dbeg[t+1])
                
        else:

            raise NotImplementedError

        if do_print: print(f'distribution simulated in {elapsed(t0)}')     

        # d. aggregate
        t0 = time.time()

        for polname in self.pols_hh:

            Outputname_hh = f'{polname.upper()}_hh_from_D'
            pol = sim.__dict__[polname]
            self.sim.__dict__[Outputname_hh] = np.sum(pol*sim.D,axis=tuple(range(1,pol.ndim)))
            
        if do_print: print(f'aggregates calculated from distribution in {elapsed(t0)}')       
        
    ############
    # 8. tests #
    ############

    test_hh_z_path = tests.hh_z_path
    test_hh_path = tests.hh_path
    test_path = tests.path
    test_jacs = tests.jacs