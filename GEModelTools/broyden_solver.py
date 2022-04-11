# contains a broyden solver for equation systems

import numpy as np

def broyden_solver(f,x0,jac,tol=1e-8,max_iter=100,backtrack_fac=0.5,max_backtrack=30,do_print=False,targets=None,unknowns=None):
    """ numerical solver using the broyden method """

    # a. initial
    x = x0.ravel()
    y = f(x)

    # b. iterate
    for it in range(max_iter):
        
        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print: 
            print(f' it = {it:3d} -> max. abs. error = {abs_diff:8.1e}')
            if not targets is None and len(targets) > 1:
                y_ = y.reshape((len(targets),-1))
                for i,target in enumerate(targets):
                    print(f'   {np.max(np.abs(y_[i])):8.1e} in {target}')

        if abs_diff < tol: return x
        
        # ii. new x
        dx = np.linalg.solve(jac,-y)
                
        # iii. evalute with backtrack
        for _ in range(max_backtrack):

            try: # evaluate
                ynew = f(x+dx)
                if np.any(np.isnan(ynew)): raise ValueError
            except ValueError: # backtrack
                if do_print: print(f'backtracking...')
                dx *= backtrack_fac
            else: # update jac and break from backtracking
                dy = ynew-y
                jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx) ** 2), dx)
                y = ynew
                x += dx
                break

        else:

            raise ValueError('Too many backtracks, maybe bad initial guess?')

    else:

        raise ValueError(f'No convergence after {max_iter} iterations')    