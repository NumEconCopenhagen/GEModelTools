import time
import numpy as np

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from copy import deepcopy

from .path import get_varnames
from consav.misc import elapsed

def hh_z_path(model):
    """ test exogenous part of household simulation along path """

    print('note: inputs = steady state value -> expected: constant value (straigt line)\n')

    par = model.par
    ss = model.ss
    path = model.path

    # a. solution and simulation hh along path
    model._set_inputs_hh_all_ss()
    model.simulate_hh_z_path(do_print=True)

    # b. show mean of z
    print('')
    fig = plt.figure(figsize=(6,4))

    Dz = path.Dz
    y = np.array([np.sum(Dz[t]*par.z_grid)for t in range(par.T)])
    y_ss = np.sum(par.z_grid*ss.Dz)

    ax = fig.add_subplot(1,1,1)
    ax.plot(np.arange(par.T),y-y_ss,'-')
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.set_title('mean(path.z[t]-ss.z)')
    ax.set_ylim([-1e-4,1e-4])

def hh_path(model):
    """ test household solution and simulation along path """

    print('note: inputs = steady state value -> expected: constant value (straigt line)\n')

    par = model.par
    ss = model.ss
    path = model.path

    # a. solution and simulation hh along path
    model._set_inputs_hh_all_ss()
    model.solve_hh_path(do_print=True)
    model.simulate_hh_path(do_print=True)

    # b. show mean of each hh output
    print('')
    fig = plt.figure(figsize=(6,len(model.outputs_hh)*4))
    for i,outputname in enumerate(model.outputs_hh):
        
        D = path.D
        pol = getattr(path,f'{outputname}')
        y = np.array([np.sum(D[t]*pol[t])for t in range(par.T)])
        y_ss = getattr(ss,f'{outputname.upper()}_hh')

        ax = fig.add_subplot(len(model.outputs_hh),1,1+i)
        ax.plot(np.arange(par.T),y-y_ss,'-')
        ax.set_yscale('symlog', linthresh=1e-8)
        Outputname_hh = f'{outputname.upper()}_hh'
        ax.set_title(f'path.{Outputname_hh}[t] - ss.{Outputname_hh}')

        ax.set_ylim([-1e-4,1e-4])
        
def print_varname_check(model,varname):

    ss = model.ss
    path = model.path

    if varname in model.targets:
        
        max_abs_val = np.max(np.abs(path.__dict__[varname][0,:]))

        print(f' {varname:15s} {max_abs_val:8.1e} [target]')

    else:

        diff = path.__dict__[varname][0,:]-ss.__dict__[varname]
        max_abs_diff = np.max(np.abs(diff))

        print(f' {varname:15s} {max_abs_diff:8.1e}')

def path(model):
    """ test evaluation of path """

    #model_ = model
    model_ = model.copy()

    par = model_.par
    ss = model_.ss
    path = model_.path

    # a. prepare
    model_._set_ini(ini_input='ss')


    # c. shock and unknowns
    inputs = []

    print('shocks: ',end='')
    for shock in model_.shocks:

        model_._set_shocks_ss()
        print(f'{shock} ',end='')
        inputs.append(shock)


    print('\nunknowns: ',end='')
    for unknown in model_.unknowns:

        model_._set_unknowns_ss()
        print(f'{unknown} ',end='')
        inputs.append(unknown)

    print('\n')

    for blockstr in model_.blocks:

        if blockstr == 'hh':

            print('hh')

            model_.solve_hh_path()
            model_.simulate_hh_path()

            for varname in model_.outputs_hh:
                varname_ = f'{varname.upper()}_hh'
                inputs.append(varname_)
                print_varname_check(model_,varname_)

        else:

            print(blockstr)

            varnames = get_varnames(blockstr)
            for varname in varnames:
                if not varname in inputs:
                    model_.path.__dict__[varname][:,:] = np.nan

            model_.call_block(blockstr)

            for varname in varnames:
                if not varname in inputs:
                    print_varname_check(model_,varname)
                    inputs.append(varname)

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