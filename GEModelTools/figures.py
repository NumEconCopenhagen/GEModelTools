import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def show_IRFs(models,labels,paths,abs_value=[],do_inputs=True,do_targets=True,T_max=None):
    
    model = models[0]
    
    par = model.par
    if T_max is None: T_max = par.transition_T
    
    # full_list
    full_list = []
    if do_inputs: full_list.append(('input, exogenous',[x for x in model.inputs_exo]))
    full_list.append(('paths',paths))
    if do_targets: full_list.append(('tagets',[x for x in model.targets]))
    
    # figures
    for (typename,varnames) in full_list:
        
        print(f'### {typename} ###')
                
        num = len(varnames)
        nrows = num//4+1
        ncols = np.fmin(num,4)
        if num%4 == 0: nrows -= 1 
            
        fig = plt.figure(figsize=(6*ncols,4*nrows))
        for i,varname in enumerate(varnames):
            
            ax = fig.add_subplot(nrows,ncols,i+1)
            ax.set_title(varname)
            
            for label,model_ in zip(labels,models):
            
                pathvalue = getattr(model_.path,varname)            
                
                if not np.isnan(getattr(model_.ss,varname)):

                    ssvalue = getattr(model_.ss,varname)

                    if varname in abs_value:
                        ax.plot(np.arange(T_max),pathvalue[:T_max]-ssvalue,label=label)
                        ax.set_ylabel('abs. diff. to of s.s.')
                    else:
                        ax.plot(np.arange(T_max),(pathvalue[:T_max]/ssvalue-1)*100,label=label)
                        ax.set_ylabel('% diff. to s.s.')

                else:

                    ax.plot(np.arange(T_max),pathvalue[:T_max],label=label)
            
            if len(labels) > 1: ax.legend(frameon=True)
            
        plt.show()
        fig.tight_layout(pad=2.0)
        print('')
