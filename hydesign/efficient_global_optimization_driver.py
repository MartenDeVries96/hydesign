# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:44:48 2023

@author: mikf
"""
import time
from openmdao.core.driver import Driver, RecordingDebugging
from hydesign.Parallel_EGO import (get_design_vars,
                                   get_xlimits,
                                   get_xtypes,
                                   get_mixint_context,
                                   get_sm,
                                   eval_sm,
                                   get_candiate_points,
                                   perturbe_around_point,
                                   extreme_around_point,
                                   cast_to_mixint,
                                   drop_duplicates,
                                   concat_to_existing,
                                   )

from six import iteritems
# from smt.utils.design_space import (
#     DesignSpace,
#     FloatVariable,
#     IntegerVariable,
#     OrdinalVariable,
# )
# from smt.applications.mixed_integer import MixedIntegerContex
# from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler#, StandardScaler

import numpy as np
from smt.sampling_methods import LHS# Random, FullFactorial
from multiprocessing import Pool
# from smt.surrogate_models import KRG, KPLS, KPLSK, GEKPLS
# from smt.applications.mixed_integer import MixedIntegerSurrogateModel
# from smt.applications.ego import Evaluator


class EGODriver(Driver):

    def __init__(self, variables, **kwargs):
        super().__init__(**kwargs)
        self.maxiter=20
        self.is_converged=False
        self.kwargs=kwargs
        self.scaler=None
        self.n_seed=0
        self.theta_bounds=[1e-06, 2e1]
        self.n_comp=4
        self.n_clusters=4
        self.npred=3e4
        self.tol=1e-6
        # self.list_vars = list_vars
        self.variables=variables
        self.opt_sign=None
        self.opt_var="NPV_over_CAPEX"
        self.n_doe=20
        self.min_conv_iter=3
        # self.design_vars = design_vars
        # self.fixed_vars = fixed_vars

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('disp', True, desc='Set to False to prevent printing')

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        if len(self._objs) > 1:
            msg = 'EGODriver currently does not support multiple objectives.'
            raise RuntimeError(msg)

        self.comm = None
    
    def model_evaluation(self, x):
        model = self._problem().model
        x = self.scaler.inverse_transform(x)
        # x_eval = self.expand_x_for_model_eval(x)

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])
        model._solve_nonlinear()
        for name, val in iteritems(self.get_objective_values()):
            obj = val
            break
        return obj
        
    # def expand_x_for_model_eval(self, x):
    
    #     list_vars = self.list_vars
    #     variables = self.variables
    #     design_vars = self.design_vars
    #     fixed_vars = self.fixed_vars
            
    #     x_eval = np.zeros([x.shape[0], len(list_vars)])
    
    #     for ii,var in enumerate(list_vars):
    #         if var in design_vars:
    #             x_eval[:,ii] = x[:,design_vars.index(var)]
    #         elif var in fixed_vars:
    #             x_eval[:,ii] = variables[var]['value']
    
    #     return x_eval


    def run(self):
        """
        Excute the stochastic gradient descent algorithm.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        model = problem.model
        model._solve_nonlinear()

        disp = self.options['disp']
        kwargs = self.kwargs

        # -----------------
        # INPUTS
        # -----------------
        
        ### paralel EGO parameters
        # n_procs = 31 # number of parallel process. Max number of processors - 1.
        # n_doe = n_procs*2
        # n_clusters = int(n_procs/2)
        #npred = 1e4
        # npred = 1e5
        # tol = 1e-6
        # min_conv_iter = 3
        
        # start_total = time.time()
        
        variables = self.variables
        design_vars, fixed_vars = get_design_vars(variables)
        xlimits = get_xlimits(variables, design_vars)
        xtypes = get_xtypes(variables, design_vars)
                
        # Scale design variables
        scaler = MinMaxScaler()
        scaler.fit(xlimits.T)
        
      
        # START Parallel-EGO optimization
        # -------------------------------------------------------        
        
        # LHS intial doe
        mixint = get_mixint_context(variables, self.n_seed)
        sampling = mixint.build_sampling_method(
          LHS, criterion="maximin", random_state=self.n_seed)
        xdoe = sampling(self.n_doe)
        xdoe = np.array(mixint.design_space.decode_values(xdoe))

        # store intial DOE
        self.xdoe = xdoe
        
        xdoe = scaler.transform(xdoe)



       # -----------------
        # HPP model
        # -----------------
        name = kwargs["name"]
        print('\n\n\n')
        print(f'Sizing a HPP plant at {name}:')
        print()
        list_minimize = ['LCOE [Euro/MWh]']
        
        # Get index of output var to optimize
        # Get sign to always write the optimization as minimize
        opt_var = self.opt_var
        opt_sign = -1
        if opt_var in list_minimize:
            opt_sign = 1
        self.opt_sign = opt_sign
        
        # kwargs['opt_sign'] = opt_sign
        # kwargs['scaler'] = scaler
        self.scaler = scaler
        kwargs['xtypes'] = xtypes
        kwargs['xlimits'] = xlimits
    
        # hpp_m = kwargs['hpp_model'](**kwargs)
        # Update kwargs to use input file generated when extracting weather
        # kwargs['input_ts_fn'] = hpp_m.input_ts_fn
        # kwargs['altitude'] = hpp_m.altitude
        # kwargs['price_fn'] = None
        
        print('\n\n')
        
        # Lists of all possible outputs, inputs to the hpp model
        # -------------------------------------------------------
        # list_vars = hpp_m.list_vars
        # list_out_vars = hpp_m.list_out_vars
        # op_var_index = list_out_vars.index(opt_var)
        # kwargs.update({'op_var_index': op_var_index})
        # Stablish types for design variables
        
        # kwargs['list_vars'] = list_vars
        # kwargs['design_vars'] = design_vars
        # kwargs['fixed_vars'] = fixed_vars
        
        # Evaluate model at initial doe
        start = time.time()
        n_procs = kwargs['n_procs']
        PE = ParallelEvaluator(n_procs = n_procs)
        ydoe = PE.run_ydoe(fun=self.evaluate_simple, x=xdoe)
        self.xdoe = xdoe
        self.ydoe = ydoe
        self.PE = PE
        
        lapse = np.round((time.time() - start)/60, 2)
        print(f'Initial {xdoe.shape[0]} simulations took {lapse} minutes')
        
        # Initialize iterative optimization
        itr = 0
        error = 1e10
        conv_iter = 0
        xopt = xdoe[[np.argmin(ydoe)],:]
        yopt = ydoe[[np.argmin(ydoe)],:]
        kwargs['yopt'] = yopt
        yold = np.copy(yopt)
        self.yold = yold
        # xold = None
        print(f'  Current solution {opt_sign}*{opt_var} = {float(yopt):.3E}'.replace('1*',''))
        print(f'  Current No. model evals: {xdoe.shape[0]}\n')
 

        # n_iter = 0
        x = xopt.copy()
        while (itr < self.maxiter) and (not self.is_converged):
            obj_value, x, error, conv_iter, success = self.objective_callback(x, itr, error, conv_iter, record=True)
            itr += 1
            if disp:
                print(itr, obj_value)
        return False
    
    def objective_callback(self, xopt, itr, error, conv_iter, record):
        success = 1

        with RecordingDebugging('EGO', self.iter_count, self) as _:
            self.iter_count += 1
            # Iteration
            start_iter = time.time()
        
            # Train surrogate model
            np.random.seed(self.n_seed)
            sm_args = {'theta_bounds': self.theta_bounds, 'n_comp': self.n_comp}
            sm = get_sm(self.xdoe, self.ydoe, **sm_args)
            self.sm = sm
            # kwargs['sm'] = sm
            
            # Evaluate surrogate model in a large number of design points
            # in parallel
            start = time.time()
            both = self.PE.run_both(self.surrogate_evaluation, itr, self.n_seed)
            # with Pool(n_procs) as p:
            #     both = ( p.map(fun_par, (np.arange(n_procs)+itr*100) * 100 + itr) )
            xpred = np.vstack([both[ii][0] for ii in range(len(both))])
            ypred_LB = np.vstack([both[ii][1] for ii in range(len(both))])
            
            # Get candidate points from clustering all sm evalautions
            n_clusters = self.n_clusters
            xnew = get_candiate_points(
                xpred, ypred_LB, 
                n_clusters = n_clusters, #n_clusters - 1, 
                quantile = 1e-2) #1/(kwargs['npred']/n_clusters) ) 
                # request candidate points based on global evaluation of current surrogate 
                # returns best designs in n_cluster of points with outputs bellow a quantile
            lapse = np.round( ( time.time() - start )/60, 2)
            print(f'Update sm and extract candidate points took {lapse} minutes')
            
            
            # -------------------
            # Refinement
            # -------------------
            # # optimize the sm starting on the cluster based candidates and the best design
            #xnew, _ = concat_to_existing(xnew, _, xopt, _)
            #xopt_iter = PE.run_xopt_iter(surrogate_optimization, xnew, **kwargs)
            
            # 2C) 
            if (np.abs(error) < self.tol): 
                #add refinement around the opt
                np.random.seed(self.n_seed*100+itr) # to have a different refinement per iteration
                step = np.random.uniform(low=0.05,high=0.25,size=1)
                xopt_iter = perturbe_around_point(xopt, step=step)
            else: 
                #add extremes on each opt_var (one at a time) around the opt
                xopt_iter = extreme_around_point(xopt)
            
            xopt_iter = self.scaler.inverse_transform(xopt_iter)
            xopt_iter = cast_to_mixint(xopt_iter, self.variables)
            xopt_iter = self.scaler.transform(xopt_iter)
            xopt_iter, _ = drop_duplicates(xopt_iter,np.zeros_like(xopt_iter))
            xopt_iter, _ = concat_to_existing(xnew,np.zeros_like(xnew), xopt_iter, np.zeros_like(xopt_iter))
        
            # run model at all candidate points
            start = time.time()
            yopt_iter = self.PE.run_ydoe(fun=self.model_evaluation, x=xopt_iter)
            
            lapse = np.round( ( time.time() - start )/60, 2)
            print(f'Check-optimal candidates: new {xopt_iter.shape[0]} simulations took {lapse} minutes')    
        
            # update the db of model evaluations, xdoe and ydoe
            xdoe_upd, ydoe_upd = concat_to_existing(self.xdoe, self.ydoe, xopt_iter, yopt_iter)
            xdoe_upd, ydoe_upd = drop_duplicates(xdoe_upd, ydoe_upd)
            
            # Drop yopt if it is not better than best design seen
            xopt = xdoe_upd[[np.argmin(ydoe_upd)],:]
            yopt = ydoe_upd[[np.argmin(ydoe_upd)],:]
            
            #if itr > 0:
            error = self.opt_sign * float(1 - (self.yold/yopt) ) 
        
            self.xdoe = np.copy(xdoe_upd)
            self.ydoe = np.copy(ydoe_upd)
            # xold = np.copy(xopt)
            self.yold = np.copy(yopt)
            itr = itr+1        
            lapse = np.round( ( time.time() - start_iter )/60, 2)
        
            print(f'  Current solution {self.opt_sign}*{self.opt_var} = {float(yopt):.3E}'.replace('1*',''))
            print(f'  Current No. model evals: {self.xdoe.shape[0]}')
            print(f'  rel_yopt_change = {error:.2E}')
            print(f'Iteration {itr} took {lapse} minutes\n')
        
            if (np.abs(error) < self.tol):
                conv_iter += 1
                if (conv_iter >= self.min_conv_iter):
                    print('Surrogate based optimization is converged.')
                    self.is_converged = True
                    # break
            else:
                conv_iter = 0
            
        return yopt, xopt, error, conv_iter, success

    def surrogate_evaluation(self, seed): # Evaluates the surrogate model
        # seed, kwargs = inputs
        mixint = get_mixint_context(self.variables, self.n_seed)
        return eval_sm(
            self.sm, mixint, 
            scaler=self.scaler,
            seed=seed, #different seed on each iteration
            npred=self.npred,
            fmin=self.yopt[0,0],)


class ParallelEvaluator:
    """
    Implement Evaluator interface using multiprocessing Pool object (Python 3 only).
    """
    def __init__(self, n_procs = 31):
        self.n_procs = n_procs
        
    def run_ydoe(self, fun, x):
        n_procs = self.n_procs
        with Pool(n_procs) as p:
            return np.array(p.map(fun, [x[[i], :] for i in range(x.shape[0])])).reshape(-1, 1)

    def run_both(self, fun, i, n_seed):
        n_procs = self.n_procs
        with Pool(n_procs) as p:
            return (p.map(fun, [(n + i * n_procs) * 100 + n_seed for n in np.arange(n_procs)]))
        
    def run_xopt_iter(self, fun, x):
        n_procs = self.n_procs
        with Pool(n_procs) as p:
            return np.vstack(p.map(fun, [x[[ii],:] for ii in range(x.shape[0])]))
