import warnings
import time
import numpy as np

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from copy import deepcopy

from .path import get_varnames
from consav.misc import elapsed

def ss(model,do_warnings=True):

    for varname in model.varlist:
    
        if np.isnan(value := model.ss.__dict__[varname]):
            print(f'{varname:15s}: nan')
            if do_warnings: warnings.warn(f'warning: {varname} contains nan') 
        else:    
            print(f'{varname:15s}: {value:12.4f}')

def hh_path(model,in_place=False,ylim=1e-4):
    """ test household solution and simulation along path """

    print('note: inputs = steady state value -> expected: constant value (straigt line) in roughly -10^-5 to 10^5\n')

    if in_place:
        model_ = model
    else:
        model_ = model.copy()

    par = model_.par
    ss = model_.ss
    path = model_.path

    # a. solution and simulation hh along path
    model_._set_inputs_hh_all_ss()
    model_.solve_hh_path(do_print=True)
    model_.simulate_hh_path(do_print=True)

    # b. show mean of each hh output
    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        print('')
        fig = plt.figure(figsize=(6,len(model_.outputs_hh)*4))
        for i,outputname in enumerate(model_.outputs_hh):
            
            D = path.D
            pol = getattr(path,f'{outputname}')
            y = np.array([np.sum(D[t]*pol[t])for t in range(par.T)])
            y_ss = getattr(ss,f'{outputname.upper()}_hh')

            ax = fig.add_subplot(len(model_.outputs_hh),1,1+i)
            ax.plot(np.arange(par.T),y-y_ss,'-')
            ax.set_yscale('symlog', linthresh=1e-8)
            Outputname_hh = f'{outputname.upper()}_hh'
            ax.set_title(f'path.{Outputname_hh}[t] - ss.{Outputname_hh}')

            ax.set_ylim([-ylim,ylim])
        
        fig.tight_layout(pad=1.0)
    
def print_varname_check(model,varname):

    ss = model.ss
    path = model.path

    if varname in model.targets:
        
        max_abs_val = np.max(np.abs(path.__dict__[varname][:,0]))

        print(f'  {varname:15s} {max_abs_val:8.1e} [target]')

    else:

        diff = path.__dict__[varname][:,0]-ss.__dict__[varname]
        max_abs_diff = np.max(np.abs(diff))

        print(f'  {varname:15s} {max_abs_diff:8.1e}')

def path(model,in_place=False,do_print=True,do_warnings=True):
    """ test evaluation of path """

    if in_place:
        model_ = model
    else:
        model_ = model.copy()

    par = model_.par
    ss = model_.ss
    path = model_.path

    # a. prepare
    model_._set_ini(ini_input='ss')

    # c. shock and unknowns
    inputs = set()

    print('shocks: ',end='')
    for shock in model_.shocks:

        model_._set_shocks_ss()
        print(f'{shock} ',end='')
        inputs.add(shock)

        if np.isnan(path.__dict__[shock]).any():
            if do_warnings: warnings.warn(f'warning: shock {shock} contains nan') 

    print('\nunknowns: ',end='')
    for unknown in model_.unknowns:

        model_._set_unknowns_ss()
        print(f'{unknown} ',end='')
        inputs.add(unknown)

        if np.isnan(path.__dict__[unknown]).any():
            if do_warnings: warnings.warn(f'warning: unknown {unknown} contains nan') 

    print('\n')
    print(f'look at max(abs(path.VARNAME[:]-ss.VARNAME)):\n')
    for blockstr in model_.blocks:

        if blockstr == 'hh':

            print(' hh')

            varnames = [inputname for inputname in  model_.inputs_hh_all]

            for varname in varnames:
                assert varname in inputs, f'{varname} not defined before hh block'
                
            model_.solve_hh_path()
            model_.simulate_hh_path()

            outputnames = [f'{outputname.upper()}_hh' for outputname in model_.outputs_hh]            
            varnames += outputnames 

        else:

            print(f' {blockstr}')

            varnames = get_varnames(blockstr)
            for varname in varnames:
                if not varname in inputs:
                    model_.path.__dict__[varname][:,:] = np.nan

            model_.call_block(blockstr)

        for varname in varnames:

            if not varname in inputs:
                print_varname_check(model_,varname)
                if np.isnan(path.__dict__[varname]).any():
                    if do_warnings: warnings.warn(f'warning: variable {varname} contains nan (in {blockstr})')     
                inputs.add(varname)
        

def jacs(model,s_list=None,dx=1e-4):
    """ test the computation of hh Jacobians with direct and fake news method, and the overall Jacobian"""

    par = model.par
    if s_list is None:
        s_list = list(np.arange(0,model.par.T,model.par.T//4))

    if len(model.outputs_hh) > 0:

        print('note: differences should only be due to numerical errors\n')

        # a. direct
        print('direct method:')
        model._compute_jac_hh(dx=dx,do_print=True,do_direct=True,s_list=s_list)
        jac_hh_direct = deepcopy(model.jac_hh)

        # b. fake news
        print(f'\nfake news method:')
        model._compute_jac_hh(dx=dx,do_print=True,do_direct=False)

        # c. compare
        fig = plt.figure(figsize=(6*2,len(model.outputs_hh)*len(model.inputs_hh_all)*4),dpi=100)

        i = 0
        for inputname in model.inputs_hh_all:
            for outputname in model.outputs_hh:
            
                jac_hh_var_direct = jac_hh_direct[(f'{outputname.upper()}_hh',inputname)]
                jac_hh_var = model.jac_hh[(f'{outputname.upper()}_hh',inputname)]
                
                ax = fig.add_subplot(len(model.inputs_hh_all)*len(model.outputs_hh),2,i*2+1)
                ax_diff = fig.add_subplot(len(model.inputs_hh_all)*len(model.outputs_hh),2,i*2+2)

                ax.set_title(f'{outputname.upper()} to {inputname}')
                ax_diff.set_title(f'... difference')

                for j,s in enumerate(s_list):
                    
                    ax.plot(np.arange(par.T),jac_hh_var_direct[:,s],color=colors[j],label=f'shock at {s}')
                    ax.plot(np.arange(par.T),jac_hh_var[:,s],color=colors[j],ls='--',label='fake news')
                    
                    diff = jac_hh_var[:,s]-jac_hh_var_direct[:,s]
                    ax_diff.plot(np.arange(par.T),diff,color=colors[j])

                if i == 0: ax.legend(frameon=True,bbox_to_anchor=(0.5,1.25))
                i += 1            

        fig.tight_layout()
        
    # e. condition numbers - full Jacobian
    print('')
    model._compute_jac(inputs='unknowns',dx=dx,do_print=True)
    model._compute_jac(inputs='shocks',dx=dx,do_print=True)


def evaluate_speed(model):
    """ test household solution and simulation along path """

    model._set_inputs_hh_all_ss()

    t0 = time.time()
    model.solve_hh_path(do_print=True)
    print(f'.solve_hh_path: {elapsed(t0)}')

    t0 = time.time()
    model.simulate_hh_path(do_print=True)    
    print(f'.simulate_hh_path: {elapsed(t0)}')

    t0 = time.time()
    model.simulate_hh_path(do_print=True)    
    print(f'.evaluate_path(): {elapsed(t0)}')