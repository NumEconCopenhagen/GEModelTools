import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams.update({'font.size':12})

from copy import deepcopy

def hh_path(self):
    """ test household solution and simulation along path """

    print('note: inputs = steady state value -> expected: constant value (straigt line)\n')

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
        y_ss = getattr(ss,outputname.upper())

        ax = fig.add_subplot(len(self.outputs_hh),1,1+i)
        ax.plot(np.arange(par.transition_T),y-y_ss,'-')
        ax.set_yscale('symlog', linthresh=1e-8)
        ax.set_title(outputname)

        ax.set_ylim([-1e-4,1e-4])
        
def path(self):
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

        print(f'{varname:15s}: t0 = {diff_t0:8.1e}, max abs. {max_abs_diff:8.1e}')

    print('\nabsolute value (potential targets):')
    for varname in self.varlist:

        pathvalue = getattr(path,varname)[0,:]
        ssvalue = getattr(ss,varname)

        if not np.isnan(ssvalue): continue
        
        max_abs = np.max(np.abs(pathvalue))

        print(f'{varname:15s}: t0 = {pathvalue[0]:8.1e}, max abs. {max_abs:8.1e}')        

def jac_hh(self,s_list,dshock=1e-6):
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