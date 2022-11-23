import time
import numpy as np

import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from copy import deepcopy

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
        
def path(model):
    """ test evaluation of path """

    print('note: inputs = steady state value -> expected: no difference to steady state and zero errors\n')

    par = model.par
    ss = model.ss
    path = model.path

    # a. set exogenous and endogenous to steady state
    model._set_shocks_ss()
    model._set_unknowns_ss()
    
    # b. baseline evaluation at steady state 
    model.evaluate_path()

    # c. 
    print('difference to value at steady state:')
    for varname in model.varlist:

        pathvalue = getattr(path,varname)[0,:]
        ssvalue = getattr(ss,varname)

        if np.isnan(ssvalue): continue

        diff_t0 = pathvalue[0]-ssvalue
        max_abs_diff = np.max(np.abs(pathvalue-ssvalue))

        print(f'{varname:15s}: t0 = {diff_t0:8.1e}, max abs. {max_abs_diff:8.1e}')

    print('\nabsolute value (potential targets):')
    for varname in model.varlist:

        pathvalue = getattr(path,varname)[0,:]
        ssvalue = getattr(ss,varname)

        if not np.isnan(ssvalue): continue
        
        max_abs = np.max(np.abs(pathvalue))

        print(f'{varname:15s}: t0 = {pathvalue[0]:8.1e}, max abs. {max_abs:8.1e}')        

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

                if i == 0: ax.legend(frameon=True)
                i += 1            

        # d. condition numbers
        print('')
        for outputname in model.outputs_hh:
            Outputname_hh = f'{outputname.upper()}_hh'
            print(f'{Outputname_hh}:')
            for inputname in model.inputs_hh_all:
                cond = np.linalg.cond(model.jac_hh[(Outputname_hh,inputname)])
                mean = np.mean(model.jac_hh[(Outputname_hh,inputname)])            
                if ~np.all(np.isclose(model.jac_hh[(Outputname_hh,inputname)],0.0)):
                    print(f' {inputname:15s}: {cond = :.1e} [{mean = :8.1e}]')
            print('')

    # e. condition numbers - full Jacobian
    model._compute_jac(inputs='unknowns',dx=dx,do_print=True)
    model._compute_jac(inputs='shocks',dx=dx,do_print=True)
    print('')

    for targetname in model.targets:
        print(f'{targetname}:')
        for inputname in model.unknowns + model.shocks:
            cond = np.linalg.cond(model.jac[(targetname,inputname)])
            mean = np.mean(model.jac[(targetname,inputname)])
            if ~np.all(np.isclose(model.jac[(targetname,inputname)],0.0)):
                print(f' {inputname:15s}: {cond = :.1e} [{mean = :8.1e}]')            
        print('')

    for jacname in ['H_U','H_Z']:
        
        jac = getattr(model,jacname)
        cond = np.linalg.cond(jac)
        mean = np.mean(jac)
        if ~np.all(np.isclose(jac,0.0)):
            print(f'{jacname:15s}: {cond = :.1e} [{mean = :8.1e}]')            

    print('')

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