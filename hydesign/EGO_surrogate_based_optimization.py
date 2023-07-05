import argparse
import glob
import os
import yaml
import time
import numpy as np
from numpy import newaxis as na
import pandas as pd
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from scipy import optimize
from scipy.stats import norm

from multiprocessing import Pool

import smt
from smt.applications.ego import EGO, Evaluator
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    FLOAT,
    ENUM,
    INT,
)
from smt.surrogate_models import KRG, KPLS, KPLSK, GEKPLS
from smt.applications.mixed_integer import MixedIntegerSurrogateModel
from smt.sampling_methods import LHS, Random, FullFactorial

from hydesign.hpp_assembly import hpp_model, mkdir
from hydesign.examples import examples_filepath

EGO_path = os.path.dirname(__file__).replace("\\", "/") + '/'

def LCB(sm, point):
    """
    Lower confidence bound optimization: minimize by using mu - 3*sigma
    """
    pred = sm.predict_values(point)
    var = sm.predict_variances(point)
    res = pred - 3.0 * np.sqrt(var)
    
    return res

def EI(sm, point, fmin=1e3):
    """
    Negative Expected improvement
    """
    pred = sm.predict_values(point)
    sig = np.sqrt(sm.predict_variances(point))
    
    args0 = (fmin - pred) / sig
    args1 = (fmin - pred) * norm.cdf(args0)
    args2 = sig * norm.pdf(args0)
    ei = args1 + args2
    return -ei


def KStd(sm, point):
    """
    Lower confidence bound optimization: minimize by using mu - 3*sigma
    """
    res = np.sqrt( sm.predict_variances(point) )
    return res

def KB(sm, point):
    """
    Mean GP process
    """
    res = sm.predict_values(point)
    return res

def get_sm(xdoe, ydoe, mixint=None):
    '''
    Function that trains the surrogate and uses it to predict on random input points
    '''    
    sm = KPLSK(
        corr="squar_exp",
        poly='linear',
        theta0=[1e-2],
        #theta_bounds=[1e-3, 1e2],
        #noise_bounds=[1e-12, 1e2],
        n_comp=4,
        print_global=False)
    sm.set_training_values(xdoe, ydoe)
    sm.train()
    
    return sm


def eval_sm(sm, mixint, scaler=None, seed=0, npred=1e3, fmin=1e10):
    '''
    Function that predicts the xepected improvement (EI) of the surrogate model based on random input points
    '''
    sampling = mixint.build_sampling_method(
        LHS, criterion="c", random_state=int(seed))
    xpred = sampling(int(npred))

    if scaler == None:
        pass
    else:
        xpred = scaler.transform(xpred)    

    ypred_LB = EI(sm=sm, point=xpred, fmin=fmin)

    return xpred, ypred_LB

def opt_sm_EI(sm, mixint, x0, fmin=1e10, n_seed=0):
    '''
    Function that optimizes the surrogate's expected improvement
    '''
    ndims = mixint.get_unfolded_dimension()
    
    func = lambda x: EI(sm, x[np.newaxis,:], fmin=fmin)
   
    minimizer_kwargs = {
        "method": "SLSQP",
        "bounds" : [(0,1)]*ndims,
        "options" : {
                "maxiter": 20,
                'eps':1e-3,
                'disp':False
            },
    }

    res = optimize.basinhopping(
        func, 
        x0 = x0, 
        niter=100, 
        stepsize=10,
        minimizer_kwargs=minimizer_kwargs,
        seed=n_seed, 
        target_accept_rate=0.5, 
        stepwise_factor=0.9)

    # res = optimize.minimize(
    #     fun = func,
    #     x0 = x0, 
    #     method="SLSQP",
    #     bounds=[(0,1)]*ndims,
    #     options={
    #         "maxiter": 100,
    #         'eps':1e-3,
    #         'disp':False
    #     },
    # )    
    
    return res.x.reshape([1,-1]) 

def opt_sm(sm, mixint, x0, fmin=1e10):
    '''
    Function that optimizes the surrogate based on lower confidence bound predictions
    '''

    ndims = mixint.get_unfolded_dimension()
    res = optimize.minimize(
        fun = lambda x:  KB(sm, x.reshape([1,ndims])),
        jac = lambda x: np.stack([sm.predict_derivatives(
           x.reshape([1,ndims]), kx=i) 
           for i in range(ndims)] ).reshape([1,ndims]),
        x0 = x0.reshape([1,ndims]),
        method="SLSQP",
        bounds=[(0,1)]*ndims,
        options={
            "maxiter": 20,
            'eps':1e-4,
            'disp':False
        },
    )
    return res.x.reshape([1,-1])

def get_candiate_points(
    x, y, quantile=0.25, n_clusters=32 ): 
    '''
    Function that groups the surrogate evaluations bellow a quantile level (quantile) and
    clusters them in n clusters (n_clusters) and returns the best input location (x) per
    cluster for acutal model evaluation
    '''

    yq = np.quantile(y,quantile)
    ind_up = np.where(y<yq)[0]
    xup = x[ind_up]
    yup = y[ind_up]
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=0,
        n_init=10,
        ).fit(xup)    
    clust_id = kmeans.predict(xup)
    xbest_per_clst = np.vstack([
        xup[np.where( yup== np.min(yup[np.where(clust_id==i)[0]]) )[0],:] 
        for i in range(n_clusters)])
    return xbest_per_clst

def extreme_around_point(x):
    ndims = x.shape[1]
    xcand = np.tile(x.T,ndims*2).T
    for i in range(ndims):
        xcand[i,i] = 0.0
    for i in range(ndims):
        xcand[i+ndims,i] = 1.0
    return xcand

def perturbe_around_point(x, step=0.1):
    ndims = x.shape[1]
    xcand = np.tile(x.T,ndims*2).T
    for i in range(ndims):
        xcand[i,i] += step
    for i in range(ndims):
        xcand[i+ndims,i] -= step
    
    xcand = np.maximum(xcand,0)
    xcand = np.minimum(xcand,1.0)
    return xcand 

def get_design_vars(variables):
    return [var_ for var_ in variables.keys() 
            if variables[var_]['var_type']=='design'
           ], [var_ for var_ in variables.keys() 
               if variables[var_]['var_type']=='fixed']

def get_limits(variables, design_var=[]):
    if len(design_var)==0:
        design_var, fixed_var = get_design_vars(variables)
    return np.array([variables[var_]['limits'] for var_ in design_var])

def get_resolution(variables, design_var=[]):
    if len(design_var)==0:
        design_var, fixed_var = get_design_vars(variables)
    return np.array([variables[var_]['resolution'] for var_ in design_var])

def round_to_resolution(x, resolution):
    xround = np.zeros_like(x)
    for ii, res in enumerate(resolution):
        xround[:,ii] = np.round(x[:,ii]/res, decimals=0)*res
    return xround

def inverse_transformed_with_resolution(u, scaler, resolution):
    x = scaler.inverse_transform(u)
    xround = round_to_resolution(x, resolution)
    return scaler.transform(xround)
    

def drop_duplicates(x,y, decimals=3):
    
    x_rounded = np.around(x, decimals=decimals)
    
    _, indices = np.unique(x_rounded, axis=0, return_index=True)
    x_unique = x[indices,:]
    y_unique = y[indices,:]
    return x_unique, y_unique

def concat_to_existing(x,y,xnew,ynew):
    x_concat, y_concat = drop_duplicates(
        np.vstack([x,xnew]),
        np.vstack([y,ynew])
        )
    return x_concat, y_concat


# class ParallelRunner():

#     def __init__(self, n_procs=None):
#         """
#         Parameters
#         ----------
#         n_procs : int or None, optional
#             Number of processes passed to multiprocessing.Pool
#         """
#         self.pool = Pool(n_procs)

#     def run(self, fun, x):
#         """Run in parallel

#         Parameters
#         ----------
#         fun : function
#             function for sequential run. Interface must be:
#         x : array
#             array of inputs to evaluate f

#         Returns
#         -------
#         results : array
#             all results
#         """

        
#         results = np.array( 
#             self.pool.map(fun, [x[[i],:] for i in range(x.shape[0])] )
#             ).reshape(-1,1)    
#         return results

    
# if __name__ == "__main__":
    
#     # -----------------------------------------------
#     # Arguments from the outer .sh (shell) script
#     # -----------------------------------------------
#     parser=argparse.ArgumentParser()
#     parser.add_argument('--example', default=None, help='ID (index) to run an example site, based on ./examples/examples_sites.csv')
#     parser.add_argument('--name', help = "Site name")
#     parser.add_argument('--longitude', help = "Site longitude")
#     parser.add_argument('--latitude', help = "Site latitude")
#     parser.add_argument('--altitude', help = "Site altitude")
#     parser.add_argument('--input_ts_fn', help = "Input ts file name")
#     parser.add_argument('--sim_pars_fn', help = "Simulation parameters file name")
#     parser.add_argument('--opt_var', help="Objective function for sizing optimization, should be one of: ['NPV_over_CAPEX','NPV [MEuro]','IRR','LCOE [Euro/MWh]','CAPEX [MEuro]','OPEX [MEuro]','penalty lifetime [MEuro]']")
#     parser.add_argument('--num_batteries', help='Maximum number of batteries to be considered in the design.')
#     parser.add_argument('--weeks_per_season_per_year', help='Number of weeks per season to be considered in the design.', default=None)
    
#     parser.add_argument('--n_procs', help='Number of processors to use')
#     parser.add_argument('--n_doe', help='Number of initial model simulations')
#     parser.add_argument('--n_clusters', help='Number of clusters to explore local vs global optima')
#     parser.add_argument('--n_seed', help='Seed number to reproduce the sampling in EGO', default=0)
#     parser.add_argument('--max_iter', help='Maximum number of parallel EGO ierations', default=10)
#     parser.add_argument('--work_dir', help='Working directory', default='./')
#     parser.add_argument('--final_design_fn', help='File name of the final design stored as csv', default=None)
    
#     args=parser.parse_args()
    
#     example = args.example
    
#     if example == None:
#         name = str(args.name)
#         longitude = int(args.longitude)
#         latitude = int(args.latitude)
#         altitude = int(args.altitude)
#         input_ts_fn = examples_filepath+str(args.input_ts_fn)
#         sim_pars_fn = examples_filepath+str(args.sim_pars_fn)
        
#     else:
#         examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
        
#         try:
#             ex_site = examples_sites.iloc[int(example),:]

#             print('Selected example site:')
#             print('---------------------------------------------------')
#             print(ex_site.T)

#             name = ex_site['name']
#             longitude = ex_site['longitude']
#             latitude = ex_site['latitude']
#             altitude = ex_site['altitude']
#             input_ts_fn = examples_filepath+ex_site['input_ts_fn']
#             sim_pars_fn = examples_filepath+ex_site['sim_pars_fn']
            
#         except:
#             raise(f'Not a valid example: {int(example)}')
    
#     opt_var = str(args.opt_var)
    
#     num_batteries = int(args.num_batteries)
#     weeks_per_season_per_year = args.weeks_per_season_per_year
#     if weeks_per_season_per_year != None:
#         weeks_per_season_per_year = int(weeks_per_season_per_year)
        
#     n_procs = int(args.n_procs)
#     n_doe = int(args.n_doe)
#     n_clusters = int(args.n_clusters)
#     n_seed = int(args.n_seed)    
#     max_iter = int(args.max_iter)
#     work_dir = str(args.work_dir) 
#     final_design_fn = str(args.final_design_fn)
        
#     if final_design_fn == None:
#         final_design_fn = f'{work_dir}design_hpp_simple_{name}_{opt_var}.csv'        
        
#     # -----------------
#     # INPUTS
#     # -----------------
    
#     ### paralel EGO parameters
#     # n_procs = 31 # number of parallel process. Max number of processors - 1.
#     # n_doe = n_procs*2
#     # n_clusters = int(n_procs/2)
#     npred = 3e4
#     tol = 1e-2
#     min_conv_iter = max_iter
#     #min_conv_iter = 3
    
#     start_total = time.time()
    
#     # -----------------
#     # HPP model
#     # -----------------
#     print('\n\n\n')
#     print(f'Sizing a HPP plant at {name}:')
#     print()
#     hpp_m = hpp_model(
#         latitude,
#         longitude,
#         altitude,
#         num_batteries = num_batteries,
#         work_dir = work_dir,
#         sim_pars_fn = sim_pars_fn,
#         input_ts_fn = input_ts_fn,
#         weeks_per_season_per_year = weeks_per_season_per_year,
#         seed = n_seed,
#     )
#     print('\n\n')
    
#     # Lists of all possible outputs, inputs to the hpp model
#     # -------------------------------------------------------
#     list_vars = hpp_m.list_vars
#     list_out_vars = hpp_m.list_out_vars
#     list_minimize = ['LCOE [Euro/MWh]']
    
#     # Get index of output var to optimize
#     op_var_index = list_out_vars.index(opt_var)
#     # Get sign to always write the optimization as minimize
#     opt_sign = -1
#     if opt_var in list_minimize:
#         opt_sign = 1
    
#     # Stablish types for design variables
#     xtypes = [
#         #FLOAT]*12
#         #clearance, sp, p_rated, Nwt, wind_MW_per_km2, 
#         INT, INT, INT, INT, FLOAT, 
#         #solar_MW, surface_tilt, surface_azimuth, DC_AC_ratio
#         INT,FLOAT,FLOAT,FLOAT,
#         #b_P, b_E_h , cost_of_battery_P_fluct_in_peak_price_ratio
#         INT,INT,FLOAT]
        

#     xlimits = np.array([
#         #clearance: min distance tip to ground
#         [10, 60],
#         #Specific Power
#         [200, 400],
#         #p_rated
#         [1, 10],
#         #Nwt
#         [0, 400],
#         #wind_MW_per_km2
#         [5, 9],
#         #solar_MW
#         [1, 400],
#         #surface_tilt
#         [0, 50],
#         #surface_azimuth
#         [150, 210],
#         #DC_AC_ratio
#         [1, 2.0],
#         #b_P in MW
#         [0, 100],
#         #b_E_h in h
#         [1, 10],
#         #cost_of_battery_P_fluct_in_peak_price_ratio
#         [0, 20],
#         ])    
    
#     # Scale design variables
#     scaler = MinMaxScaler()
#     scaler.fit(xlimits.T)
    
#     # Create a parallel evaluator of the model
#     # -------------------------------------------------------
#     def fun(x): 
#         try:
#             x = scaler.inverse_transform(x)
#             return np.array(
#                 opt_sign*hpp_m.evaluate(*x[0,:])[op_var_index])
#         except:
#             print( ( 'x='+', '.join(str(x).split()) ).replace('[[','[').replace(']]',']') )
 
#     class ParallelEvaluator(Evaluator):
#         """
#         Implement Evaluator interface using multiprocessing Pool object (Python 3 only).
#         """
#         def __init__(self, n_procs = 31):
#             self.n_procs = n_procs
            
#         def run(self, fun, x):
#             n_procs = self.n_procs
#             # Caveat: import are made here due to SMT documentation building process
#             import numpy as np
#             from sys import version_info
#             from multiprocessing import Pool

#             if version_info.major == 2:
#                 raise('version_info.major==2')
                
#             # Python 3 only
#             with Pool(n_procs) as p:
#                 return np.array(
#                     p.map(fun, [x[[i],:] for i in range(x.shape[0])] ) 
#                 ).reshape(-1,1)
    
#     # START Parallel-EGO optimization
#     # -------------------------------------------------------        
    
#     # LHS intial doe
#     mixint = MixedIntegerContext(xtypes, xlimits)
#     sampling = mixint.build_sampling_method(
#       LHS, criterion="ese", random_state=n_seed)
#     xdoe = sampling(n_doe)
#     xdoe = scaler.transform(xdoe)
    
#     #     # Full factorial doe to test all vertices
#     #     mixint = MixedIntegerContext(xtypes, xlimits)
#     #     sampling = FullFactorial(xlimits=np.array([[0,1]]*12) )
#     #     xdoe = sampling(int(n_doe*0.75))

#     #     # LHS doe
#     #     ndoe_add = n_doe - int(n_doe*0.75)
#     #     sampling = mixint.build_sampling_method(
#     #         LHS, criterion="ese", random_state=n_seed)
#     #     xdoe_add = sampling(ndoe_add)
#     #     xdoe_add = scaler.transform(xdoe_add)

#     #     xdoe = np.vstack([xdoe,xdoe_add])


#     # Evaluate model at initial doe
#     start = time.time()
#     ydoe = ParallelEvaluator(
#         n_procs = n_procs).run(fun=fun,x=xdoe)
        
#     lapse = np.round((time.time() - start)/60, 2)
#     print(f'Initial {xdoe.shape[0]} simulations took {lapse} minutes')
    
#     # Initialize iterative optimization
#     itr = 0
#     error = 1e10
#     conv_iter = 0
#     xopt = xdoe[[np.argmin(ydoe)],:]
#     yopt = ydoe[[np.argmin(ydoe)],:]
#     yold = np.copy(yopt)
#     xold = None
#     print(f'  Current solution {opt_sign}*{opt_var} = {float(yopt):.3E}\n'.replace('1*',''))
    
#     while itr < max_iter:
#         # Iteration
#         start_iter = time.time()

#         # -------------------
#         # Train surrogate model
#         # -------------------
#         np.random.seed(n_seed)
#         sm = get_sm(xdoe, ydoe, mixint)
        
#         # --------------------------------
#         # Get good candidates based on EI
#         # --------------------------------
        
#         # Option A
#         # Evaluate EI of surrogate model in a large number of points in parallel 
#         start = time.time()
#         def fun_par(seed): return eval_sm(
#             sm, mixint, 
#             scaler=scaler,
#             seed=seed, #different seed on each iteration
#             npred=npred,
#             fmin=yopt[0,0],
#         )
#         with Pool(n_procs) as p:
#             both = ( p.map(fun_par, (np.arange(n_procs)+n_procs*itr)* 100 + n_seed ) )
#         xpred = np.vstack([both[ii][0] for ii in range(len(both))])
#         ypred_LB = np.vstack([both[ii][1] for ii in range(len(both))])

#         # Get candidate points from clustering all sm evalautions
#         xnew = get_candiate_points(
#            xpred, ypred_LB, 
#            n_clusters = n_clusters, 
#            quantile = 1e-2) 

#         # # Option B
#         # # request candidate points based on global optimization of surrogate's EI 
#         # sampling = mixint.build_sampling_method(
#         #       LHS, criterion="c", random_state=n_seed)
#         # xnew_0 = sampling(n_clusters)
#         # xnew_0 = scaler.transform(xnew_0)
#         # def fun_opt(x): 
#         #     return opt_sm_EI(sm, mixint, x, fmin=yopt[0,0], n_seed=n_seed)
#         # with Pool(n_procs) as p:
#         #     xnew = np.vstack(
#         #             p.map(fun_opt, [xnew_0[ii,:] 
#         #             for ii in range(xnew_0.shape[0])] )  
#         #         )
            
#         lapse = np.round( ( time.time() - start )/60, 2)
#         print(f'Update sm and extract candidate points took {lapse} minutes')
        
#         # -------------------
#         # Refinement
#         # -------------------
        
#         # 2A) based on SM. Surrogate believer.
#         # # optimize the sm starting on the cluster based candidates 
#         # xnew, _ = concat_to_existing(xnew, np.zeros_like(xnew), xopt, np.zeros_like(xopt))
#         # def fun_opt(x): 
#         #     return opt_sm(sm, mixint, x, fmin=yopt[0,0])
#         # with Pool(n_procs) as p:
#         #     xopt_iter = np.vstack(
#         #             p.map(fun_opt, [xnew[[ii],:] 
#         #             for ii in range(xnew.shape[0])] ) 
#         #         )
        
#         # 2B) based on EI
#         # # optimize the sm starting on the cluster based candidates 
#         # xnew, _ = concat_to_existing(xnew, np.zeros_like(xnew), xopt, np.zeros_like(xopt))
#         # def fun_opt(x): 
#         #     return opt_sm_EI(sm, mixint, x, fmin=yopt[0,0])
#         # with Pool(n_procs) as p:
#         #     xopt_iter = np.vstack(
#         #             p.map(fun_opt, [xnew[[ii],:] 
#         #             for ii in range(xnew.shape[0])] ) 
#         #         )

#         # 2C) add refinement around the opt
#         if (np.abs(error) < tol):
#             xopt_iter = perturbe_around_point(xopt, step=0.1)
#         else: #add extremes around the opt
#             xopt_iter = extreme_around_point(xopt)
        
#         # Doen't help
#         #xopt_iter_B = perturbe_around_point(xopt, step=0.1)
#         #xopt_iter, _ = concat_to_existing(xopt_iter, np.zeros_like(xopt_iter), xopt_iter_B, np.zeros_like(xopt_iter_B))
        
        
#         xopt_iter = scaler.inverse_transform(xopt_iter)
#         xopt_iter = np.array([mixint.cast_to_mixed_integer( xopt_iter[i,:]) 
#                        for i in range(xopt_iter.shape[0])]).reshape(xopt_iter.shape)
#         xopt_iter = scaler.transform(xopt_iter)
#         xopt_iter, _ = drop_duplicates(xopt_iter,np.zeros_like(xopt_iter))
#         xopt_iter, _ = concat_to_existing(xnew,np.zeros_like(xnew), xopt_iter, np.zeros_like(xopt_iter))
        
#         # 2D) No refinement
#         # xopt_iter, _ = drop_duplicates(xnew,np.zeros_like(xnew))

#         # run model at all candidate points
#         start = time.time()
#         yopt_iter = ParallelEvaluator(
#           n_procs = n_procs).run(fun=fun,x=xopt_iter)
        
#         lapse = np.round( ( time.time() - start )/60, 2)
#         print(f'Check-optimal candidates: new {xopt_iter.shape[0]} simulations took {lapse} minutes')    

#         # update the db of model evaluations, xdoe and ydoe
#         xdoe_upd, ydoe_upd = concat_to_existing(xdoe,ydoe, xopt_iter,yopt_iter)
#         xdoe_upd, ydoe_upd = drop_duplicates(xdoe_upd, ydoe_upd)
        
#         # Drop yopt if it is not better than best design seen
#         xopt = xdoe_upd[[np.argmin(ydoe_upd)],:]
#         yopt = ydoe_upd[[np.argmin(ydoe_upd)],:]
        
#         #if itr > 0:
#         error = opt_sign * float(1 - (yold/yopt) ) 
#         print(f'  Current solution {opt_sign}*{opt_var} = {float(yopt):.3E}')
#         print(f'  rel_yopt_change = {error:.2E}'.replace('1*',''))

#         xdoe = np.copy(xdoe_upd)
#         ydoe = np.copy(ydoe_upd)
#         xold = np.copy(xopt)
#         yold = np.copy(yopt)
#         itr = itr+1

#         lapse = np.round( ( time.time() - start_iter )/60, 2)
#         print(f'Iteration {itr} took {lapse} minutes\n')

#         if (np.abs(error) < tol):
#             conv_iter += 1
#             if (conv_iter >= min_conv_iter):
#                 print('Surrogate based optimization is converged.')
#                 break
#         else:
#             conv_iter = 0
    
#     xopt = scaler.inverse_transform(xopt)
    
#     # Re-Evaluate the last design to get all outputs
#     outs = hpp_m.evaluate(*xopt[0,:])
#     yopt = np.array(opt_sign*outs[[op_var_index]])[:,na]
#     print()
#     print(f'Objective function: {opt_var} \n')
#     print()
#     hpp_m.print_design(xopt[0,:], outs)

#     n_model_evals = xdoe.shape[0] 
    
#     lapse = np.round( ( time.time() - start_total )/60, 2)
#     print(f'Optimization with {itr} iterations and {n_model_evals} model evaluations took {lapse} minutes\n')

#     # Store results
#     # -----------------
#     design_df = pd.DataFrame(columns = list_vars, index=[name])
#     for iv, var in enumerate(list_vars):
#         design_df[var] = xopt[0,iv]
#     for iv, var in enumerate(list_out_vars):
#         design_df[var] = outs[iv]
    
#     design_df['design obj'] = opt_var
#     design_df['opt time [min]'] = lapse
#     design_df['n_model_evals'] = n_model_evals
    
#     design_df.T.to_csv(final_design_fn)
