# contains functions for plotting

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams.update({'font.size':12})

def show_IRFs(models,labels,paths,
            abs_value=None,lvl_value=None,facs=None,
            do_inputs=True,do_targets=True,ncols=4,T_max=None,
            filename=None):
    
    abs_value = [] if abs_value is None else abs_value
    lvl_value = [] if lvl_value is None else lvl_value
    facs = {} if facs is None else facs

    for path in paths: 
        if not path in facs: facs[path] = 1.0

    model = models[0]
    
    par = model.par
    if T_max is None: T_max = par.transition_T
    
    # full_list
    full_list = []
    if do_inputs: full_list.append(('inputs, exogenous',[x for x in model.inputs_exo]))
    full_list.append(('paths',paths))
    if do_targets: full_list.append(('tagets',[x for x in model.targets]))
    
    # figures
    for (typename,varnames) in full_list:
        
        print(f'### {typename} ###')
        
        num = len(varnames)
        nrows = num//ncols+1
        if num%ncols == 0: nrows -= 1 
            
        fig = plt.figure(figsize=(6*ncols,4*nrows),dpi=100)
        for i,varname in enumerate(varnames):
            
            ax = fig.add_subplot(nrows,ncols,i+1)
            ax.set_title(varname,fontsize=14)
            
            for label,model_ in zip(labels,models):
            
                pathvalue = getattr(model_.path,varname)[0,:]            
                
                if not np.isnan(getattr(model_.ss,varname)):

                    ssvalue = getattr(model_.ss,varname)

                    if varname in abs_value:

                        ax.plot(np.arange(T_max),facs[varname]*(pathvalue[:T_max]-ssvalue),label=label)
                        if varname in facs:
                            ax.set_ylabel(fr'{facs[varname]:.0f} x abs. diff. to of s.s.')
                        else:
                            ax.set_ylabel('abs. diff. to of s.s.')

                    elif varname in lvl_value:

                        ax.plot(np.arange(T_max),facs[varname]*pathvalue[:T_max],label=label)
                        if not np.isclose(facs[varname],1.0):
                            ax.set_ylabel(fr'{facs[varname]:.0f} x ')
                        else:
                            ax.set_ylabel('')

                    else:
                        ax.plot(np.arange(T_max),(pathvalue[:T_max]/ssvalue-1)*100,label=label)
                        ax.set_ylabel('% diff. to s.s.')

                else:

                    ax.plot(np.arange(T_max),pathvalue[:T_max],label=label)
            
            if len(labels) > 1 and i == 0: ax.legend(frameon=True)
            
        plt.show()
        fig.tight_layout(pad=3.0)
        print('')

        # save
        if not filename is None: fig.savefig(filename)
