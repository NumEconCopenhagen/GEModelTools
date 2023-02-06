# contains main GEModelClass

import importlib
import time

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from EconModel import jit
from consav.misc import elapsed

from . import tests
from . import DAG

from .simulate_hh import find_i_and_w_1d_1d, find_i_and_w_1d_1d_path, find_i_and_w_2d_1d, find_i_and_w_2d_1d_path
from .simulate_hh import simulate_hh_forwards_endo, simulate_hh_forwards_exo
from .simulate_hh import simulate_hh_forwards_endo_transpose, simulate_hh_forwards_exo_transpose
from .simulate_hh import simulate_hh_ss, simulate_hh_path

from .broyden_solver import broyden_solver
from .simulate import update_IRF_hh,simulate_agg,simulate_agg_hh
from .figures import show_IRFs
from .path import get_varnames


class GEModelClass:

    ############
    # 1. setup #
    ############

    def info(self,only_blocks=False,ss=False):
        """ print info about the model """

        par = self.par

        if not only_blocks:
            
            print('settings:')

            print(f' {par.py_hh = }')
            print(f' {par.py_block = }')
            print(f' {par.full_z_trans = }')
            print(f' {par.T = }')

            print('\nhouseholds:')
            for attr in ['grids_hh','pols_hh','inputs_hh','inputs_hh_z','outputs_hh','intertemps_hh']:
                varstr = f' {attr}: ['
                for el in getattr(self,attr):
                    varstr += f'{el},'
                if not varstr[-1] == '[':
                    varstr = varstr[:-1] + ']'
                else:
                    varstr += ']'
                print(varstr)

            print('\naggregate:')
            for attr in ['shocks','unknowns','targets']:
                varstr = f' {attr}: ['
                for el in getattr(self,attr):
                    varstr += f'{el},'
                if not varstr[-1] == '[':
                    varstr = varstr[:-1] + ']'
                else:
                    varstr += ']'
                print(varstr)

            print('\nblocks (inputs -> outputs):')

        all_inputs = self.shocks+self.unknowns
        for blockstr in self.blocks:
            
            if blockstr == 'hh':
                blockname = 'hh'
                varnames = self.inputs_hh_all + [f'{varname.upper()}_hh' for varname in self.outputs_hh]
            else:
                blockname = blockstr.split('.')[1]
                varnames = get_varnames(blockstr)
            
            inputs = [varname for varname in varnames if varname in all_inputs] 
            outputs = [varname for varname in varnames if varname not in all_inputs]

            if only_blocks:
                print(f'{blockname}:',end='')
            else:
                print(f' {blockname}:',end='')

            inputsstr = ' ['
            for varname in inputs:
                inputsstr += f'{varname}'
                if ss:
                    inputsstr += f'={getattr(self.ss,varname):.2f},'
                else:
                    inputsstr += ','

            if not inputsstr[-1] == '[':
                inputsstr = inputsstr[:-1] + ']'
            else:
                inputsstr += ']'

            print(inputsstr,end='')

            outputsstr = ' -> ['
            for varname in outputs:
                outputsstr += f'{varname}'
                if ss:
                    outputsstr += f'={getattr(self.ss,varname):.2f},'
                else:
                    outputsstr += ','                
            
            if not outputsstr[-1] == '[':
                outputsstr = outputsstr[:-1] + ']'
            else:
                outputsstr += ']'

            print(outputsstr)   

            all_inputs += outputs

    def validate_blocks(self):
        """ validate list of blocks """

        # a. types
        assert type(self.blocks) is list, '.blocks should be list'

        for blockstr in self.blocks:

            assert type(blockstr) is str, f'.blocks should be list[str]'
            
            msg = f'{blockstr}, elements in .blocks should have format MODULE.FUNCTION or be hh'
            assert blockstr == 'hh' or len(blockstr.split('.')) == 2, msg
            
        # b. check inputs
        for blockstr in self.blocks:
            
            if blockstr == 'hh': continue

            modulename,funcname = blockstr.split('.')
            
            try:
                module = importlib.import_module(modulename)
            except:
                print(f'{blockstr = }')
                raise

            assert hasattr(module,funcname), f'function {funcname} does not exist in {modulename}'
            
            func = eval(f'module.{funcname}')
            
            assert func.__code__.co_varnames[0] == 'par', f'1st argument of {blockstr} should be par'
            assert func.__code__.co_varnames[1] == 'ini', f'2nd argument of {blockstr} should be ini'
            assert func.__code__.co_varnames[2] == 'ss', f'3rd argument of {blockstr} should be ss'

    def get_varlist_from_blocks(self,blocks=None):
        """ get variable list from list of blocks """

        if blocks is None: blocks = self.blocks

        varlist = []

        for blockstr in blocks:
            
            if blockstr == 'hh':
                varnames = self.inputs_hh + self.inputs_hh_z + [f'{varname.upper()}_hh' for varname in self.outputs_hh]
            else:
                varnames = get_varnames(blockstr)

            for varname in varnames:
                if varname not in varlist: varlist.append(varname)

        return varlist
    
    def allocate_GE(self,update_varlist=True,update_hh=True,ss_nan=True):
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

        # blocks
        assert hasattr(self,'blocks'), f'.blocks must be defined'

        self.validate_blocks()
        if update_varlist: self.varlist = self.get_varlist_from_blocks()

        # varlists
        varlist_attrs = [
            'grids_hh','pols_hh','inputs_hh','inputs_hh_z','outputs_hh','intertemps_hh',
            'shocks','unknowns','targets','varlist','blocks',
        ]

        for varlist in varlist_attrs:

            assert hasattr(self,varlist), f'.{varlist} must be defined, list[str]'
            strlist = getattr(self,varlist)
            assert type(strlist) is list, f'.{varlist} must be a list, but is {type(strlist)}'
            for el in varlist:
                assert type(el) is str, f'.{varlist} must be a list of strings'

        # functions
        assert hasattr(self,'solve_hh_backwards'), f'.solve_hh_backwards must be defined'

        # saved
        for attr in varlist_attrs + ['jac','H_U','H_Z','jac_hh','IRF']:
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
            if not Varname_hh in self.varlist:
                self.varlist.append(Varname_hh)
            
        for varname in self.shocks + self.unknowns + self.targets:
            assert varname in self.varlist, f'{varname} not in .varlist'

        restricted_varnames = ['z_trans','D','pol_indices','pol_weights']
        
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
        par.__dict__.setdefault('py_hh',True)
        par.__dict__.setdefault('py_block',True)
        par.__dict__.setdefault('full_z_trans',False)

        # c. allocate grids and transition matrices
        if update_hh:

            for varname in self.grids_hh:

                gridname = f'{varname}_grid' 
                Nx = par.__dict__[f'N{varname}']
                gridarray = np.zeros(Nx)
                setattr(par,gridname,gridarray)
                
            par.z_grid = np.zeros(par.Nz)

        # d. allocate household variables
        Npol = len(self.pols_hh)
        path_pol_shape = (par.T,*sol_shape)
        sim_pol_shape = (par.simT,*sol_shape)

        if update_hh:
            
            # i. ss and path
            for varname in self.outputs_hh + self.intertemps_hh:
                ss.__dict__[varname] = np.zeros(sol_shape)
                path.__dict__[varname] = np.zeros(path_pol_shape)

            ini.Dbeg = np.zeros(sol_shape)

            if par.full_z_trans:
                ss.z_trans = np.zeros((par.Nfix,*sol_shape_endo,par.Nz,par.Nz))
            else:
                ss.z_trans = np.zeros((par.Nfix,par.Nz,par.Nz))

            ss.D = np.zeros(sol_shape)
            ss.Dbeg = np.zeros(sol_shape)
            ss.pol_indices = np.zeros((Npol,*sol_shape),dtype=np.int_)
            ss.pol_weights = np.zeros((Npol,*sol_shape))

            if par.full_z_trans:
                path.z_trans = np.zeros((par.T,par.Nfix,*sol_shape_endo,par.Nz,par.Nz))
            else:
                path.z_trans = np.zeros((par.T,par.Nfix,par.Nz,par.Nz))
                
            path.D = np.zeros(path_pol_shape)
            path.Dbeg = np.zeros(path_pol_shape)
            path.pol_indices = np.zeros((par.T,Npol,*sol_shape),dtype=np.int_)
            path.pol_weights = np.zeros((par.T,Npol,*sol_shape))

            # ii. sim
            for polname in self.pols_hh:
                sim.__dict__[polname] = np.zeros(sim_pol_shape)
                sim.__dict__[f'{polname.upper()}_hh_from_D'] = np.zeros(par.simT)

            sim.z_trans = np.zeros((par.simT,par.Nfix,par.Nz,par.Nz))
            sim.D = np.zeros(sim_pol_shape)
            sim.Dbeg = np.zeros(sim_pol_shape)
            sim.pol_indices = np.zeros((par.simT,Npol,*sol_shape),dtype=np.int_)
            sim.pol_weights = np.zeros((par.simT,Npol,*sol_shape))

        # e. allocate path and sim variables
        path_shape = (par.T,1)
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

    def update_aggregate_settings(self,shocks=None,unknowns=None,targets=None):
        """ update aggregate settings and re-allocate jac etc. """
        
        if not shocks is None: self.shocks = shocks
        if not unknowns is None: self.unknowns = unknowns
        if not targets is None: self.targets = targets

        self.allocate_GE(update_varlist=False,update_hh=False,ss_nan=False)

    def _check_z_trans(self,z_trans):
        """ check transition maxtrix """

        par = self.par

        if par.full_z_trans:
            
            pass
            # for i_fix in range(par.Nfix):
            #     for i_a in range(par.Na):
            #         for i_z in range(par.Nz):
            #             rowsum = np.sum(z_trans[i_fix,i_a,i_z,:])
            #             check = np.isclose(rowsum,1.0)
            #             assert check, f'sum(ss.z_trans[{i_fix},{i_a},{i_z},:] = {rowsum:12.8f}, should be 1.0'
        
        else:

            for i_fix in range(par.Nfix):
                for i_z in range(par.Nz):
                    rowsum = np.sum(z_trans[i_fix,i_z,:])
                    check = np.isclose(rowsum,1.0)
                    assert check, f'sum(ss.z_trans[{i_fix},{i_z},:] = {rowsum:12.8f}, should be 1.0'

    def _check_z_trans_ss(self):
        """ check steady state transition maxtrix """

        self._check_z_trans(self.ss.z_trans)

    ## functions ###
    
    def reload_hh_function(self,attr):
        """ reload household function """

        func = getattr(self,attr)
        module = importlib.import_module(f'{func.__module__}')
        func_reloaded = eval(f'module.{func.__name__}')
        setattr(self,attr,func_reloaded)

        return func_reloaded

    def call_hh_function(self,funcname,inputs):
        """ call household function """

        func = self.reload_hh_function(funcname)

        if self.par.py_hh:
            if hasattr(func,'py_func'):
                func.py_func(**inputs)
            else:
                func(**inputs)
        else:
            assert hasattr(func,'py_func'), f'function {funcname} is not decorated with @nb.njit (add or try par.py_hh = True)'
            func(**inputs)

    def call_block(self,blockstr):
        """ call block function"""

        modulename,funcname = blockstr.split('.')

        # a. inputs
        varnames = get_varnames(blockstr)
        inputvars = {varname:getattr(self.path,varname) for varname in varnames}

        # b. reload
        module = importlib.import_module(modulename)
        func = eval(f'module.{funcname}')
        
        # c. call
        with jit(self,show_exc=False) as model:
            if self.par.py_block:
                if hasattr(func,'py_func'):
                    func.py_func(model.par,model.ini,model.ss,**inputvars)
                else:
                    func(model.par,model.ini,model.ss,**inputvars)
            else:
                assert hasattr(func,'py_func'), f'{blockstr} is not decorated with @nb.njit (add or try par.py_blocks = True)'
                func(model.par,model.ini,model.ss,**inputvars)

    ###################
    # 2. steady state #
    ###################

    def _find_i_and_w_dict(self,ss_dict):
        """ find policy indices and weights from dict """

        par = self.par

        if len(self.grids_hh) == 1:

            pol1 = ss_dict[f'{self.grids_hh[0]}']
            grid1 = getattr(par,f'{self.grids_hh[0]}_grid')
            find_i_and_w_1d_1d(pol1,grid1,ss_dict['pol_indices'][0],ss_dict['pol_weights'][0])

        elif len(self.grids_hh) == 2 and len(self.pols_hh) == 2:

            pol1 = ss_dict[f'{self.grids_hh[0]}']
            pol2 = ss_dict[f'{self.grids_hh[1]}']
            grid1 = getattr(par, f'{self.grids_hh[0]}_grid')
            grid2 = getattr(par, f'{self.grids_hh[1]}_grid')
            find_i_and_w_2d_1d(pol1,grid1,grid1,grid2,ss_dict['pol_indices'][0],ss_dict['pol_weights'][0])
            find_i_and_w_2d_1d(pol2,grid2,grid1,grid2,ss_dict['pol_indices'][1],ss_dict['pol_weights'][1])
        
        else:

            raise NotImplementedError

    def _find_i_and_w_ss(self):
        """ find policy indices and weights in steady state """

        self._find_i_and_w_dict(self.ss.__dict__)

    def _get_stepvars_hh_ss(self,outputs_inplace=True):
        """ get variables for backwards step in steady state """

        par = self.par
        ss = self.ss

        # a. inputs
        stepvars_hh = {'par':par}
        for varname in self.inputs_hh_all: stepvars_hh[varname] = getattr(ss,varname)
        for varname in self.intertemps_hh: stepvars_hh[f'{varname}_plus'] = getattr(ss,varname)

        # b. outputs
        for varname in self.outputs_hh + self.intertemps_hh: 
            if outputs_inplace:
                stepvars_hh[varname] = getattr(ss,varname)
                stepvars_hh['z_trans'] = ss.z_trans
            else:
                stepvars_hh[varname] = getattr(ss,varname).copy()
                stepvars_hh['z_trans'] = ss.z_trans.copy()

        return stepvars_hh

    def _get_stepvars_hh_z_ss(self):
        """ get variables for transition matrix in steady state """

        par = self.par
        ss = self.ss

        stepvars_hh = {'par':par,'z_trans':ss.z_trans}
        for varname in self.inputs_hh_z: stepvars_hh[varname] = getattr(ss,varname)

        return stepvars_hh

    def set_hh_initial_guess(self):
        """ set initial guess for household policy functions """

        for varname in self.inputs_hh_all:
            ssvalue = getattr(self.ss,varname)
            assert np.isfinite(ssvalue), f'invalid value {varname} = {ssvalue}'

        # single evaluation
        with jit(self,show_exc=False) as model:
            
            par = model.par
            ss = model.ss

            stepvars_hh = self._get_stepvars_hh_ss()
            stepvars_hh['ss'] = True
            self.call_hh_function('solve_hh_backwards',stepvars_hh)

        # checks
        for varname in self.outputs_hh:
            assert np.all(np.isfinite(getattr(self.ss,varname))), f'invalid values in ss.{varname}'

        self._check_z_trans_ss()

    def solve_hh_ss(self,do_print=False,initial_guess=None):
        """ solve the household problem in steady state """

        t0 = time.time()

        if initial_guess is None: initial_guess = {}

        # a. prepare model to find steady state
        self.prepare_hh_ss()

        # check
        for varname in self.inputs_hh_all:
            ssvalue = getattr(self.ss,varname)
            assert np.isfinite(ssvalue), f'invalid value {varname} = {ssvalue}'

        for varname in self.pols_hh:
            assert np.all(np.isfinite(getattr(self.ss,varname))), f'invalid values in ss.{varname} (initial guess)'
        
        self._check_z_trans_ss()

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
                self.call_hh_function('solve_hh_backwards',stepvars_hh)

                # iii. check change in policy
                max_abs_diff = max([np.max(np.abs(getattr(ss,pol)-old[pol])) for pol in self.pols_hh])
                if max_abs_diff < par.tol_solve: 
                    break
                
                # iv. increment
                it += 1
                if it > par.max_iter_solve: 
                    raise ValueError('solve_hh_ss(), too many iterations')

        if do_print: print(f'household problem in ss solved in {elapsed(t0)} [{it} iterations]')

        # checks
        for varname in self.outputs_hh:
            assert np.all(np.isfinite(getattr(self.ss,varname))), f'invalid values in ss.{varname}'

        self._check_z_trans_ss()

        # c. indices and weights    
        self._find_i_and_w_ss()

    def simulate_hh_ss(self,do_print=False,Dbeg=None,find_i_and_w=False):
        """ simulate the household problem in steady state """
        
        par = self.par
        ss = self.ss

        # required: all inputs in .inputs_hh_all, ss.z_trans and all policies in .pols_hh

        t0 = time.time()

        for varname in self.pols_hh:
            assert np.all(np.isfinite(getattr(self.ss,varname))), f'invalid values in ss.{varname}'        

        self._check_z_trans_ss()

        if find_i_and_w: self._find_i_and_w_ss()

        # a. initial guess
        if not Dbeg is None:
            ss.Dbeg[:] = Dbeg

        # check
        Dbeg_sum = np.sum(ss.Dbeg)
        assert np.isclose(Dbeg_sum,1.0), f'sum(ss.Dbeg) = {Dbeg_sum:12.8f}, should be 1.0'
        
        # b. simulate
        with jit(self,show_exc=False) as model:            
            it = simulate_hh_ss(model.par,model.ss)

        # c. aggregate
        for outputname in self.outputs_hh:

            pol = ss.__dict__[outputname]
            ssvalue = np.sum(pol*ss.D)

            Outputname_hh = f'{outputname.upper()}_hh'
            ss.__dict__[Outputname_hh] = ssvalue

        # checks
        assert np.all(np.isfinite(ss.D)), f'invalid values in ss.D'

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
            find_i_and_w_1d_1d_path(par.T,path_pol1,grid1,path.pol_indices[:,0],path.pol_weights[:,0])

        elif len(self.grids_hh) == 2 and len(self.pols_hh) == 2:

            path_pol1 = getattr(path,f'{self.grids_hh[0]}')
            path_pol2 = getattr(path,f'{self.grids_hh[1]}')
            grid1 = getattr(par, f'{self.grids_hh[0]}_grid')
            grid2 = getattr(par, f'{self.grids_hh[1]}_grid')
            find_i_and_w_2d_1d_path(par.T,path_pol1,grid1,grid1,grid2,path.pol_indices[:,0],path.pol_weights[:,0])
            find_i_and_w_2d_1d_path(par.T,path_pol2,grid2,grid1,grid2,path.pol_indices[:,1],path.pol_weights[:,1])   

        else:

            raise NotImplemented

    def _get_stepvars_hh_path(self,t):
        """ get variables for backwards step in along transition path"""

        par = self.par
        path = self.path
        ss = self.ss

        stepvars_hh = {'par':par,'z_trans':path.z_trans[t]}
        for varname in self.inputs_hh_all: stepvars_hh[varname] = getattr(path,varname)[t,0]
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

        # checks
        for varname in self.inputs_hh_all:
            assert np.all(np.isfinite(getattr(self.path,varname))), f'invalid values in path.{varname}'        

        with jit(self,show_exc=False) as model:

            par = model.par
            ss = model.ss
            path = model.path
            
            for k in range(par.T):

                t = (par.T-1)-k
                if len(self.inputs_hh_z) == 0:
                    path.z_trans[t] = ss.z_trans

                # ii. stepbacks
                stepvars_hh = self._get_stepvars_hh_path(t)
                self.call_hh_function('solve_hh_backwards',stepvars_hh)

            assert np.all(np.isfinite(getattr(self.path,'z_trans'))), f'invalid values in path.z_trans'

        # c. indices and weights
        self._find_i_and_w_path()

        # checks
        for varname in self.outputs_hh:
            assert np.all(np.isfinite(getattr(self.path,varname))), f'invalid values in path.{varname}'

        if do_print: print(f'household problem solved along transition path in {elapsed(t0)}')

    def simulate_hh_path(self,do_print=False,Dbeg=None):
        """ simulate the household problem along the transition path"""
        
        par = self.par
        ss = self.ss
        path = self.path

        t0 = time.time() 

        for varname in self.pols_hh:
            assert np.all(np.isfinite(getattr(self.path,varname))), f'invalid values in path.{varname}'  

        # a. initial distribution
        if Dbeg is None:
            path.Dbeg[0] = self.ss.Dbeg
        else:
            path.Dbeg[0] = Dbeg

        # check
        Dbeg_sum = np.sum(path.Dbeg[0])
        assert np.isclose(Dbeg_sum,1.0), f'sum(path.Dbeg[0]) = {Dbeg_sum:12.8f}, should be 1.0'

        # b. simulate
        with jit(self,show_exc=False) as model:
            simulate_hh_path(model.par,model.path)

        # c. aggregate
        for outputname in self.outputs_hh:

            Outputname_hh = f'{outputname.upper()}_hh'
            pathvalue = path.__dict__[Outputname_hh]

            pol = path.__dict__[outputname]
            pathvalue[:,0] = np.sum(pol*path.D,axis=tuple(range(1,pol.ndim)))
            
        # checks
        assert np.all(np.isfinite(path.D)), f'invalid values in path.D'

        if do_print: print(f'household problem simulated along transition in {elapsed(t0)}')

    def _set_inputs_hh_all_ss(self):
        """ set household inputs to steady state """

        for inputname in self.inputs_hh_all:

            ssvalue = getattr(self.ss,inputname)
            patharray = getattr(self.path,inputname)
            patharray[:,:] = ssvalue

    def decompose_hh_path(self,do_print=False,Dbeg=None,use_inputs=None,custom_paths=None,fix_z_trans=False):
        """ decompose household transition path wrt. inputs or initial distribution """

        par = self.par
        ss = self.ss

        if use_inputs is None: 
            use_inputs = []
        elif use_inputs == 'all':
            use_inputs = self.inputs_hh_all
        else:
            use_inputs = use_inputs

        if custom_paths is None: custom_paths = {}

        # a. save original path and create clean path
        path_original = self.path
        path = self.path = deepcopy(self.path)

        for varname in self.varlist:
            if varname in self.inputs_hh_all: continue 
            path.__dict__[varname][:] = np.nan

        # b. set inputs
        for varname in self.inputs_hh_all:
            if varname in use_inputs: continue
            path.__dict__[varname][:,0] = ss.__dict__[varname]
        
        for varname in use_inputs:
            if varname in custom_paths:
                path.__dict__[varname][:,0] = custom_paths[varname]

        # c. solve and simulate
        if not use_inputs == 'all': self.solve_hh_path(do_print=do_print)

        if fix_z_trans:
            for t in range(par.T):
                path.z_trans[t] = ss.z_trans

        self.simulate_hh_path(do_print=do_print,Dbeg=Dbeg)

        # d. aggregates
        for outputname in self.outputs_hh:

            Outputname_hh = f'{outputname.upper()}_hh'
            pathvalue = path.__dict__[Outputname_hh]

            pol = path.__dict__[outputname]
            pathvalue[:,0] = np.sum(pol*path.D,axis=tuple(range(1,pol.ndim)))
            
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

    def _set_unknowns(self,x,inputs):
        """ set unknowns """

        par = self.par
        path = self.path

        Ninputs = len(inputs)

        if x.size > Ninputs*par.T:

            x = x.reshape((Ninputs,par.T,-1))
            for i,varname in enumerate(inputs):
                array = getattr(path,varname)                    
                array[:,:] = x[i,:,:]

        else:

            if not x is None:
                x = x.reshape((len(inputs),par.T))
                for i,varname in enumerate(inputs):
                    array = getattr(path,varname)                    
                    array[:,0] = x[i,:]

    def _get_errors(self,inputs=None):
        """ get errors from targets """
        
        if not inputs is None:

            errors = np.zeros((len(self.targets),self.par.T,len(inputs)*self.par.T))
            for i,varname in enumerate(self.targets):
                errors[i,:,:] = getattr(self.path,varname)[:,:]

        else:

            errors = np.zeros((len(self.targets),self.par.T))
            for i,varname in enumerate(self.targets):
                if hasattr(self,'_target_values'):
                    errors[i,:] = getattr(self.path,varname)[:,0] - self._target_values[varname].ravel()
                else:
                    errors[i,:] = getattr(self.path,varname)[:,0]

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
            shockarray[-1,0] += dx

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
            self.call_hh_function('solve_hh_backwards',one_step_ss)

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

                for polname in self.outputs_hh:
                    self.dpols[(polname,inputname)] = np.zeros(getattr(path,polname).shape)

                for s in range(par.T):
                    
                    # i. solve gradually backwards
                    stepvars_hh = self._get_stepvars_hh_ss(outputs_inplace=False)

                    do_z_trans = (s == 0) and (len(self.inputs_hh_z) > 0) and (inputname in self.inputs_hh_z)
                    
                    if s == 0:
                        
                        stepvars_hh[inputname] += dx

                    else:

                        for varname in self.intertemps_hh:
                            varname_plus = f'{varname}_plus'
                            stepvars_hh[varname_plus] = stepvars_hh[varname_plus] + dintertemps[varname]

                    self.call_hh_function('solve_hh_backwards',stepvars_hh)

                    for varname in self.intertemps_hh:
                        dintertemps[varname] = stepvars_hh[varname]-one_step_ss[varname]

                    for polname in self.outputs_hh:
                        self.dpols[(polname,inputname)][s] = (stepvars_hh[polname]-one_step_ss[polname])/dx

                    if do_z_trans:

                        D0 = np.zeros(ss.D.shape)
                        z_trans = stepvars_hh['z_trans']
                        simulate_hh_forwards_exo(ss.Dbeg,z_trans,D0)

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
            curly_E[outputname][0] = simulate_hh_forwards_exo_transpose(temp,ss.z_trans)

        for t in range(1,par.T-1):
            
            for outputname in self.outputs_hh:
                temp = simulate_hh_forwards_endo_transpose(curly_E[outputname][t-1],ss.pol_indices,ss.pol_weights)
                curly_E[outputname][t] = simulate_hh_forwards_exo_transpose(temp,ss.z_trans)
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

    def _compute_jac(self,inputs=None,dx=1e-4,do_print=False):
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
        self.evaluate_blocks(use_jac_hh=True)
        evaluate_t += time.time()-t0_

        base = self._get_errors().ravel() 
        path_ss = deepcopy(path)
        
        # c. calculate
        for varname in self.varlist:
            path.__dict__[varname] = np.zeros((par.T,len(inputs)*par.T))

        self._set_shocks_ss()
        self._set_unknowns_ss()

        if do_unknowns:
            jac_mat = self.H_U
        elif do_shocks:
            jac_mat = self.H_Z
        else:
            jac_dict = {}
        
        x_ss = np.zeros((len(inputs),par.T))
        for i,varname in enumerate(inputs):
            x_ss[i,:] = getattr(self.ss,varname)

        # i. inputs
        x0 = np.zeros((x_ss.size,x_ss.size))
        for i in range(x_ss.size):   

            x0[:,i] = x_ss.ravel().copy()
            x0[i,i] += dx

        # ii. evaluate
        self._set_unknowns(x0,inputs)
        t0_ = time.time()
        self.evaluate_blocks(ncols=len(inputs)*par.T,use_jac_hh=True)
        evaluate_t += time.time()-t0_
        errors = self._get_errors(inputs)

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
                    jac[:,s] = (path.__dict__[outputname][:,thread]-path_ss.__dict__[outputname][:,0])/dx

        if do_print:
            if do_unknowns:
                print(f'full Jacobian to unknowns computed in {elapsed(t0)} [in evaluate_blocks(): {elapsed(0,evaluate_t)}]')
            else: 
                print(f'full Jacobian to shocks computed in {elapsed(t0)} [in evaluate_blocks(): {elapsed(0,evaluate_t)}]')

        # reset
        self.path = path_original

        if not (do_unknowns or do_shocks):
            return jac_dict
            
    def compute_jacs(self,dx=1e-4,skip_hh=False,inputs_hh_all=None,skip_shocks=False,do_print=False,do_direct=False):
        """ compute all Jacobians """
        
        if not skip_hh and len(self.outputs_hh) > 0:
            if do_print: print('household Jacobians:')
            self._compute_jac_hh(inputs_hh_all=inputs_hh_all,dx=dx,do_direct=do_direct,do_print=do_print)
            if do_print: print('')

        if do_print: print('full Jacobians:')
        self._compute_jac(inputs='unknowns',dx=dx,do_print=do_print)
        if not skip_shocks: self._compute_jac(inputs='shocks',dx=dx,do_print=do_print)

    ####################################
    # 5. find transition path and IRFs #
    ####################################

    def _check_shocks(self,shocks):
        """ check shock specification """

        par = self.par

        if shocks is None: raise ValueError('shocks must be specified: 1) list[VARNAME], 2) {dVARNAME:PATH}')

        if type(shocks) is str and not shocks == 'all': raise ValueError("shocks must be list or 'all' ")
        
        if type(shocks) is list:

            for shock in shocks:
                assert shock in self.shocks, f'shocks have element {shock}, not in .shocks = {self.shocks}'

        elif type(shocks) is dict:

            for shock,values in shocks.items():

                assert shock[0] == 'd', f'shocks have element {shock}, must have format dVARNAME'
                assert shock[1:] in self.shocks, f'shocks have element {shock}, but {shock[1]} not in .shocks = {self.shocks}'
                assert values.size == par.T, f'the values for shocks element {shock} have size {values.size}, must be {par.T = }'

        return shocks

    def _set_shocks(self,shocks='all',std_shock=False):
        """ set shocks as AR(1) or from detailed specification """
        
        Tshock = self.par.T//2

        # a. AR(1)s
        if type(shocks) is list or shocks == 'all': 

            for shockname in self.shocks:

                patharray = getattr(self.path,shockname)
                ssvalue = getattr(self.ss,shockname)

                # i. ss-value
                patharray[:,:] = ssvalue 

                if type(shocks) is list and not shockname in shocks: continue
                
                # ii. jump and rho
                if std_shock:
                    
                    stdname = f'std_{shockname}'

                    if hasattr(self.par,stdname):
                        scale = getattr(self.par,stdname)
                    else:
                        patharray[:,:] = ssvalue         
                        continue                    

                    assert not scale < 0, f'{stdname} must not be negative'

                else:

                    jumpname = f'jump_{shockname}'

                    if hasattr(self.par,jumpname):
                        scale = getattr(self.par,jumpname)
                    else:
                        patharray[:,:] = ssvalue         
                        continue
                
                rhoname = f'rho_{shockname}'
                rho = getattr(self.par,rhoname)

                # iii. shock value
                patharray[:Tshock,0] += scale*rho**np.arange(Tshock)

        # b. shock specificaiton
        else:

            # i. validate
            for k,v in shocks.items():

                if not k[0] == 'd':                    
                    raise ValueError(f'error for {k} in shocks, format should be {{dVARNAME:PATH}}')
                if not k[1:] in self.shocks: 
                    raise ValueError(f'error for {k} in shocks, format should be {{dVARNAME:PATH}} with VARNAME in .shocks')
                if not v.ravel().size == self.par.T: 
                    raise ValueError(f'shocks[k].shape = {shocks[k].shape}, should imply .ravel().size == par.T')

            # ii. apply
            for shockname in self.shocks:

                patharray = getattr(self.path,shockname)
                ssvalue = getattr(self.ss,shockname)
                
                # a. custom path
                patharray[:,:] = ssvalue
                if (dshockname := f'd{shockname}') in shocks:
                    patharray[:Tshock,0] = ssvalue + shocks[dshockname][:Tshock].ravel()

    def _set_ini(self,ini_input=None):
        """ set initial distribution """

        par = self.par
        ini = self.ini
        ss = self.ss

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
            
    def find_IRFs(self,shocks=None,reuse_G_U=False,do_print=False):
        """ find linearized impulse responses """
        
        shocks = self._check_shocks(shocks)

        par = self.par
        ss = self.ss
        path = self.path

        path_original = deepcopy(path)

        t0 = time.time()

        # a. solution matrix
        t0_ = time.time()
        if not reuse_G_U: self.G_U[:,:] = -np.linalg.solve(self.H_U,self.H_Z)       
        t1_ = time.time()
        
        # b. set path for shocks
        self._set_shocks(shocks=shocks)

        # c. IRFs

        # shocks
        dZ = np.zeros((len(self.shocks),par.T))
        for i_shock,shockname in enumerate(self.shocks):
            dZ[i_shock,:] = path.__dict__[shockname][:,0]-ss.__dict__[shockname]
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

        # reset
        self.path = path_original

        if do_print: print(f'linear transition path found in {elapsed(t0)} [finding solution matrix: {elapsed(t0_,t1_)}]')

    def evaluate_blocks(self,ini='ss',ncols=1,use_jac_hh=False):
        """ evaluate transition path """

        par = self.par
        ss = self.ss
        path = self.path

        assert use_jac_hh or ncols == 1
        
        # a. update initial distribution
        self._set_ini(ini_input=ini)

        # b. evaluate each blove
        for blockstr in self.blocks:

            # i. household block
            if blockstr == 'hh':

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

                            pathvalue[:,:] += (jac_hh@(pathvalue_input-ssvalue_input)) 
                
                elif len(self.outputs_hh) > 0: # non-linear solution
                    
                    # i. solve
                    self.solve_hh_path()

                    # ii. simulate
                    self.simulate_hh_path(Dbeg=self.ini.Dbeg)

            else:

                self.call_block(blockstr)
            
    def _evaluate_H(self,x,do_print=False):
        """ compute error in equation system for targets """
        
        # a. evaluate
        self._set_unknowns(x,self.unknowns)
        self.evaluate_blocks(ini=None)
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

    def find_transition_path(self,shocks,ini='ss',unknowns_ss=True,do_end_check=True,do_print=False,do_print_unknowns=False):
        """ find transiton path (fully non-linear) """

        par = self.par
        ss = self.ss
        path = self.path

        shocks = self._check_shocks(shocks)

        t0 = time.time()

        # a. set path for shocks
        self._set_shocks(shocks=shocks)

        # b. set initial value of unknowns to ss
        x0 = np.zeros((len(self.unknowns),par.T))
        for i,varname in enumerate(self.unknowns):
            if unknowns_ss:
                x0[i,:] = getattr(self.ss,varname)
            else:
                x0[i,:] = getattr(self.path,varname)[:,0]

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
        obj(x)

        # f. test
        if do_end_check:
            for varname in self.varlist:
                ssval = ss.__dict__[varname]
                if np.isnan(ssval): continue
                endpathval = path.__dict__[varname][-1,0]
                if not np.isclose(ssval,endpathval):
                    print(f'{varname}: terminal value is {endpathval:12.8f}, but ss value is {ssval:12.8f}')

        if do_print: print(f'\ntransition path found in {elapsed(t0)}')

    def decompose_blocks(self,shocks,blocks,labels=None,unknowns=None,targets=None,do_print=False,do_plot=False,do_end_check=False,**kwargs):
        """ decompose sub-block """

        par = self.par
        ss = self.ss

        # a1. validate input format
        if unknowns is None: unknowns = []
        if targets is None: targets = [{}]*len(shocks)

        msg = f'shocks must be list of dict, [{{dVARNAME:VALUES}}], with same keys'
        assert type(shocks) is list, msg
        shocks_str = [k[1:] for k in shocks[0].keys()]
        for shocks_ in shocks:
            assert type(shocks_) is dict, msg
            assert [k[1:] for k in shocks_.keys()] == shocks_str, msg

        msg = f'targets must be list of dict, [{{dVARNAME:VALUES}}], with same keys'
        assert type(targets) is list, msg
        targets_str =  [k for k in targets[0].keys()]
        for targets_ in targets:
            assert type(targets_) is dict, msg
            assert  [k for k in targets_.keys()] == targets_str, msg
        assert len(shocks) == len(targets)

        assert len(targets_str) == len(unknowns), f'number of unknowns = {len(unknowns)} must equal number of targets = {len(targets_str)}' 

        # a2. validate input names
        varlist = self.get_varlist_from_blocks(blocks)
        
        for unknown in unknowns:
            assert unknown in varlist, f'unknown {unknown} is not in varlist of blocks, {varlist}'

        for shock in shocks_str:
            assert shock in varlist, f'shocks key shock is not in varlist of blocks, {varlist}'

        for target in targets_str:
            assert target in varlist, f'target {target} is not in varlist of blocks, {varlist}'

        # a. base copy
        model_ = self.copy()

        model_.unknowns = unknowns
        model_.targets = targets_str
        model_.shocks = [x for x in self.varlist if not x in unknowns+targets] + shocks_str
        model_.blocks = blocks

        # b. re-compute Jacobian
        model_.allocate_GE(update_varlist=False,update_hh=False,ss_nan=False)

        # c. find transition paths
        paths = []
        models = []
        for shocks_,targets_ in zip(shocks,targets):

            if len(targets_str) == 0:
                model_._check_shocks(shocks_)
                model_._set_shocks(shocks=shocks_)
                model_.evaluate_blocks()
            else:
                model_._target_values = targets_
                model_.compute_jacs(do_print=do_print,skip_hh=True,skip_shocks=True)
                model_.find_transition_path(do_print=do_print,shocks=shocks_,do_end_check=do_end_check)

            model__ = SimpleNamespace()
            model__.par = self.par
            model__.ss = self.ss
            model__.IRF = self.IRF
            model__.shocks = model_.shocks
            model__.unknowns = model_.unknowns
            model__.targets = model_.targets
            model__.path = deepcopy(model_.path)
            models.append(model__)

            paths.append(model_.path)


        # d. plot
        if do_plot:
            if labels is None: labels = [None]*len(models)
            show_IRFs(models,labels,varlist,do_shocks=False,do_targets=False,**kwargs)

        return paths

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

    def _get_stepvars_hh_z_sim(self,t):
        """ get variables for transition matrix in simulation """

        par = self.par
        sim = self.sim
        ss = self.ss

        stepvars_hh_z = {'par':par,'z_trans':sim.z_trans[t]}
        for varname in self.inputs_hh_z: stepvars_hh_z[varname] = getattr(ss,varname) + getattr(sim,f'd{varname}')[t]

        return stepvars_hh_z

    def prepare_simulate(self,skip_hh=False,only_pols_hh=True,reuse_G_U=False,do_print=True):
        """ prepare model for simulation by calculating IRFs """

        par = self.par
        ss = self.ss
        path = self.path

        varlist_hh = self.pols_hh if only_pols_hh else self.outputs_hh

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
        for i_shock,shockname in enumerate(self.shocks):
            
            # i. shocks
            dZ = np.zeros((len(self.shocks),par.T))        
            dZ[i_shock,:] = path.__dict__[shockname][:,0]-ss.__dict__[shockname]

            self.IRF[(shockname,shockname)][:] = dZ[i_shock,:] 

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
                for polname in varlist_hh:
                    IRF_pols = self.IRF['pols'][(polname,shockname)] = np.zeros((*ss.D.shape,par.T))
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

    def simulate(self,do_prepare=True,skip_hh=False,only_pols_hh=True,reuse_G_U=False,do_print=False):
        """ simulate the model """

        par = self.par
        ss = self.ss
        sim = self.sim

        varlist_hh = self.pols_hh if only_pols_hh else self.outputs_hh

        # a. prepare simulation
        if do_prepare: self.prepare_simulate(skip_hh=skip_hh,only_pols_hh=only_pols_hh,reuse_G_U=reuse_G_U,do_print=do_print)

        t0 = time.time()

        # a. IRF matrix
        IRF_mat = np.zeros((len(self.varlist),len(self.shocks),par.T))
        for i,varname in enumerate(self.varlist):
            for j,shockname in enumerate(self.shocks):
                IRF = self.IRF[(varname,shockname)]
                if np.any(np.isnan(IRF)):
                    IRF_mat[i,j,:] = 0.0
                else:
                    IRF_mat[i,j,:] = IRF

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
            IRF_pols_mat = np.zeros((len(varlist_hh),len(self.shocks),*ss.D.shape,par.T))
            for i,polname in enumerate(varlist_hh):
                for j,shockname in enumerate(self.shocks):
                    IRF_pols_mat[i,j] = self.IRF['pols'][(polname,shockname)]

            sim_pols_mat = np.zeros((len(varlist_hh),par.simT,*ss.D.shape))
            
            t0_ = time.time()
            simulate_agg_hh(epsilons,IRF_pols_mat,sim_pols_mat)
            t1_ = time.time()

            for i,polname in enumerate(varlist_hh):
                sim_pols_mat[i] += ss.__dict__[polname]
                sim.__dict__[polname] = sim_pols_mat[i]

            if do_print: print(f'household policies simulated in {elapsed(t0)}')

            # ii. distribution
            self.simulate_distribution(do_print=do_print)

    def simulate_distribution(self,only_pols_hh=True,do_print=False):
        """ simulate distribution """

        par = self.par
        ss = self.ss
        sim = self.sim

        varlist_hh = self.pols_hh if only_pols_hh else self.outputs_hh

        t0 = time.time()

        # a. initialize
        sim.Dbeg[0] = ss.Dbeg
        
        # b. simulate
        if len(self.grids_hh) == 1:

            grid1 = getattr(par,f'{self.grids_hh[0]}_grid')

            for t in range(par.simT):

                if len(self.inputs_hh_z) > 0:
                    raise NotImplementedError('timing varying z_trans not implemented')
                else:
                    sim.z_trans[t,:] = ss.z_trans

                simulate_hh_forwards_exo(sim.Dbeg[t],sim.z_trans[t],sim.D[t])    
                
                if t < par.simT-1:

                    sim_i = sim.pol_indices[t]
                    sim_w = sim.pol_weights[t]
                    sim_pol = sim.__dict__[self.pols_hh[0]]
                    find_i_and_w_1d_1d(sim_pol[t],grid1,sim_i[0],sim_w[0])
                    simulate_hh_forwards_endo(sim.D[t],sim_i,sim_w,sim.Dbeg[t+1])
                
        else:

            raise NotImplementedError

        if do_print: print(f'distribution simulated in {elapsed(t0)}')     

        # c. aggregate
        t0 = time.time()

        for polname in varlist_hh:

            Outputname_hh = f'{polname.upper()}_hh_from_D'
            pol = sim.__dict__[polname]
            sim.__dict__[Outputname_hh] = np.sum(pol*sim.D,axis=tuple(range(1,pol.ndim)))
            
        if do_print: print(f'aggregates calculated from distribution in {elapsed(t0)}')       
        
    ############
    # 8. tests #
    ############

    test_hh_z_path = tests.hh_z_path
    test_hh_path = tests.hh_path
    test_path = tests.path
    test_jacs = tests.jacs
    test_evaluate_speed = tests.evaluate_speed

    ##########
    # 9. DAG #
    ##########

    draw_DAG = DAG.draw_DAG