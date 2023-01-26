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

from hydesign.hpp_assembly_simplified import hpp_model_simple, mkdir
from hydesign.examples import examples_filepath

import os
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
    Expected improvement
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
    sm = KRG(
        corr="squar_exp",
        poly='linear',
        theta0=[1e-2],
        theta_bounds=[1e-2, 1e2],
        print_global=False)

    # # surrogate = KRG( 
    # #     corr="squar_exp",
    # #     theta0=[1e-2],
    # #     theta_bounds=[1e-2, 1e2],
    # #     n_start=5,
    # #     print_global=False)

    # # xlimits = mixint._xlimits
    # # xtypes = mixint._xtypes
    # # sm = MixedIntegerSurrogateModel(
    # #     categorical_kernel=smt.applications.mixed_integer.GOWER,
    # #     xtypes=xtypes,
    # #     xlimits=xlimits,
    # #     surrogate=surrogate,
    # # )

    sm.set_training_values(xdoe, ydoe)
    sm.train()
    
    return sm


def eval_sm(sm, mixint, scaler=None, seed=0, npred=1e3, fmin=1e10):
    '''
    Function that predicts the xepected improvement (EI) of the surrogate model based on random input points
    '''
    ndims = mixint.get_unfolded_dimension()
    npred = int(npred)
    
    np.random.seed(int(seed))
    sampling = mixint.build_sampling_method(Random)
    xpred = sampling(npred)
    
    if scaler == None:
        pass
    else:
        xpred = scaler.transform(xpred)
        
    #ypred_LB = LCB(sm=sm, point=xpred)
    ypred_LB = EI(sm=sm, point=xpred, fmin=fmin)

    return xpred, ypred_LB

def opt_sm(sm, mixint, x0, fmin=1e10):
    '''
    Function that optimizes the surrogate based on lower confidence bound predictions
    '''
    ndims = mixint.get_unfolded_dimension()
    res = optimize.minimize(
        #fun = lambda x:  EI(sm, x.reshape([1,ndims]), fmin=fmin)[0,0],
        fun = lambda x:  LCB(sm, x.reshape([1,ndims]))[0,0],
        x0 = x0.reshape([1,ndims]),
        method="SLSQP",
        #bounds=mixint.get_unfolded_xlimits(),
        bounds=[(0,1)]*ndims,
        options={
            "maxiter": 200,
            'eps':1e-3,
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
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=0).fit(xup)    
    clust_id = kmeans.predict(xup)
    xbest_per_clst = np.vstack([
        xup[np.where( yup== np.min(yup[np.where(clust_id==i)[0]]) )[0],:] 
        for i in range(n_clusters)])
    return xbest_per_clst

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


class ParallelRunner():

    def __init__(self, n_procs=None):
        """
        Parameters
        ----------
        n_procs : int or None, optional
            Number of processes passed to multiprocessing.Pool
        """
        self.pool = multiprocessing.Pool(n_procs)

    def run(self, fun, x):
        """Run in parallel

        Parameters
        ----------
        fun : function
            function for sequential run. Interface must be:
        x : array
            array of inputs to evaluate f

        Returns
        -------
        results : array
            all results
        """

        
        results = np.array( 
            self.pool.map(fun, [x[[i],:] for i in range(x.shape[0])] )
            ).reshape(-1,1)    
        return results

    
if __name__ == "__main__":
    
    # -----------------------------------------------
    # Arguments from the outer .sh (shell) script
    # -----------------------------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument('--example', default=None, help='ID (index) to run an example site, based on ./examples/examples_sites.csv')
    parser.add_argument('--name', help = "Site name")
    parser.add_argument('--longitude', help = "Site longitude")
    parser.add_argument('--latitude', help = "Site latitude")
    parser.add_argument('--altitude', help = "Site altitude")
    parser.add_argument('--input_ts_fn', help = "Input ts file name")
    parser.add_argument('--sim_pars_fn', help = "Simulation parameters file name")
    parser.add_argument('--opt_var', help="Objective function for sizing optimization, should be one of: ['NPV_over_CAPEX','NPV [MEuro]','IRR','LCOE [Euro/MWh]','CAPEX [MEuro]','OPEX [MEuro]','penalty lifetime [MEuro]']")
    parser.add_argument('--rotor_diameter_m', help='WT rotor diameter [m]')
    parser.add_argument('--hub_height_m', help='WT hub height [m]')
    parser.add_argument('--wt_rated_power_MW', help='WT rated power [MW]')
    parser.add_argument('--surface_tilt_deg', help='PV surface tilt [deg]')
    parser.add_argument('--surface_azimuth_deg', help='PV surface azimuth [deg]')
    parser.add_argument('--DC_AC_ratio', help='PV DC/AC ratio, this ratio defines how much overplanting of DC power is done with respect the inverter. P_DC/P_AC [-]')
    parser.add_argument('--num_batteries', help='Maximum number of batteries to be considered in the design.')
    
    parser.add_argument('--n_procs', help='Number of processors to use')
    parser.add_argument('--n_doe', help='Number of initial model simulations')
    parser.add_argument('--n_clusters', help='Number of clusters to explore local vs global optima')
    parser.add_argument('--n_seed', help='Seed number to reproduce the sampling in EGO', default=0)
    parser.add_argument('--max_iter', help='Maximum number of parallel EGO ierations', default=10)
    
    parser.add_argument('--final_design_fn', help='File name of the final design stored as csv', default=None)
    
    args=parser.parse_args()
    
    example = args.example
    
    if example == None:
        name = str(args.name)
        longitude = int(args.longitude)
        latitude = int(args.latitude)
        altitude = int(args.altitude)
        input_ts_fn = examples_filepath+str(args.input_ts_fn)
        sim_pars_fn = examples_filepath+str(args.sim_pars_fn)
        
    else:
        examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
        
        try:
            ex_site = examples_sites.iloc[int(example),:]

            print('Selected example site:')
            print('---------------------------------------------------')
            print(ex_site.T)

            name = ex_site['name']
            longitude = ex_site['longitude']
            latitude = ex_site['latitude']
            altitude = ex_site['altitude']
            input_ts_fn = examples_filepath+ex_site['input_ts_fn']
            sim_pars_fn = examples_filepath+ex_site['sim_pars_fn']
            
        except:
            raise(f'Not a valid example: {int(example)}')
    
    opt_var = str(args.opt_var)
        
    rotor_diameter_m = float(args.rotor_diameter_m)
    hub_height_m = float(args.hub_height_m)
    wt_rated_power_MW = float(args.wt_rated_power_MW)
    surface_tilt_deg = float(args.surface_tilt_deg)
    surface_azimuth_deg = float(args.surface_azimuth_deg)
    DC_AC_ratio = float(args.DC_AC_ratio)
    num_batteries = int(args.num_batteries)
    
    n_procs = int(args.n_procs)
    n_doe = int(args.n_doe)
    n_clusters = int(args.n_clusters)
    n_seed = int(args.n_seed)    
    max_iter = int(args.max_iter)
    final_design_fn = str(args.final_design_fn)
        
    work_dir = './'
    if final_design_fn == None:
        final_design_fn = f'{work_dir}design_hpp_simple_{name}_{opt_var}.csv'        
        
    # -----------------
    # INPUTS
    # -----------------
    
    ### paralel EGO parameters
    # n_procs = 31 # number of parallel process. Max number of processors - 1.
    # n_doe = n_procs*2
    # n_clusters = int(n_procs/2)
    #npred = 1e4
    npred = 1e5
    tol = 1e-6
    min_conv_iter = 3
    
    start_total = time.time()
    
    # -----------------
    # HPP model
    # -----------------
    print('\n\n\n')
    print(f'Sizing a HPP plant at {name}:')
    print()
    hpp_m = hpp_model_simple(
            latitude,
            longitude,
            altitude,
            rotor_diameter_m = rotor_diameter_m,
            hub_height_m = hub_height_m,
            wt_rated_power_MW = wt_rated_power_MW,
            surface_tilt_deg = surface_tilt_deg,
            surface_azimuth_deg = surface_azimuth_deg,
            DC_AC_ratio = DC_AC_ratio,
            num_batteries = num_batteries,
            work_dir = work_dir,
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
    )
    
    print('\n\n')
    
    # Lists of all possible outputs, inputs to the hpp model
    # -------------------------------------------------------
    list_vars = hpp_m.list_vars
    list_out_vars = hpp_m.list_out_vars
    list_minimize = ['LCOE [Euro/MWh]']
    
    # Get index of output var to optimize
    op_var_index = list_out_vars.index(opt_var)
    # Get sign to always write the optimization as minimize
    opt_sign = -1
    if opt_var in list_minimize:
        opt_sign = 1
    
    # Stablish types for design variables
    xtypes = [
        #Nwt, wind_MW_per_km2, 
        INT, FLOAT, 
        #solar_MW
        INT,
        #b_P, b_E_h , cost_of_battery_P_fluct_in_peak_price_ratio
        INT,INT,FLOAT]

    xlimits = np.array([
        #Nwt
        [0, 500],
        #wind_MW_per_km2
        [5, 9],
        #solar_MW
        [0, 500],
        #b_P in MW
        [0, 300],
        #b_E_h in h
        [1, 24],
        #cost_of_battery_P_fluct_in_peak_price_ratio
        [0, 20],
        ])    
    
    # Scale design variables
    scaler = MinMaxScaler()
    scaler.fit(xlimits.T)
    
    # Create a parallel evaluator of the model
    # -------------------------------------------------------
    def fun(x): 
        try:
            x = scaler.inverse_transform(x)
            return np.array(
                opt_sign*hpp_m.evaluate(*x[0,:])[op_var_index])
        except:
            print( ( 'x='+', '.join(str(x).split()) ).replace('[[','[').replace(']]',']') )
 
    class ParallelEvaluator(Evaluator):
        """
        Implement Evaluator interface using multiprocessing Pool object (Python 3 only).
        """
        def __init__(self, n_procs = 31):
            self.n_procs = n_procs
            
        def run(self, fun, x):
            n_procs = self.n_procs
            # Caveat: import are made here due to SMT documentation building process
            import numpy as np
            from sys import version_info
            from multiprocessing import Pool

            if version_info.major == 2:
                raise('version_info.major==2')
                
            # Python 3 only
            with Pool(n_procs) as p:
                return np.array(
                    p.map(fun, [x[[i],:] for i in range(x.shape[0])] ) 
                ).reshape(-1,1)
    
    # START Parallel-EGO optimization
    # -------------------------------------------------------        
    
    # LHS intial doe
    mixint = MixedIntegerContext(xtypes, xlimits)
    sampling = mixint.build_sampling_method(
      LHS, criterion="maximin", random_state=n_seed)
    xdoe = sampling(n_doe)
    xdoe = scaler.transform(xdoe)

    # Evaluate model at initial doe
    start = time.time()
    ydoe = ParallelEvaluator(
        n_procs = n_procs).run(fun=fun,x=xdoe)
    
    lapse = np.round((time.time() - start)/60, 2)
    print(f'Initial {xdoe.shape[0]} simulations took {lapse} minutes\n')
    
    # Initialize iterative optimization
    itr = 0
    error = 1e10
    conv_iter = 0
    yopt = ydoe[[np.argmin(ydoe)],:]
    yold = np.copy(yopt)
    xold = None
    while itr < max_iter:
        # Iteration
        start_iter = time.time()

        # Train surrogate model
        np.random.seed(n_seed)
        sm = get_sm(xdoe, ydoe, mixint)
        
        # Evaluate surrogate model in a large number of design points
        # in parallel
        start = time.time()
        def fun_par(seed): return eval_sm(
            sm, mixint, 
            scaler=scaler,
            seed=seed*100+itr, #different seed on each iteration
            npred=npred,
            fmin=yopt[0,0],
        )
        with Pool(n_procs) as p:
            both = ( p.map(fun_par, np.arange(n_procs)+itr*100 ) )
        xpred = np.vstack([both[ii][0] for ii in range(len(both))])
        ypred_LB = np.vstack([both[ii][1] for ii in range(len(both))])
        
        # Get candidate points from clustering all sm evalautions
        xnew = get_candiate_points(
            xpred, ypred_LB, 
            n_clusters = n_clusters, 
            quantile = 1/(npred/n_clusters) ) 
            # request candidate points based on global evaluation of current surrogate 
            # returns best designs in n_cluster of points with outputs bellow a quantile
        lapse = np.round( ( time.time() - start )/60, 2)
        print(f'Update sm and extract candidate points took {lapse} minutes')
        
        # # optimize the sm starting on the cluster based candidates 
        def fun_opt(x): 
            return opt_sm(sm, mixint, x, fmin=yopt[0,0])
        with Pool(n_procs) as p:
            xopt_iter = np.vstack(
                    p.map(fun_opt, [xnew[[ii],:] 
                    for ii in range(xnew.shape[0])] ) 
                )
        
        xopt_iter = scaler.inverse_transform(xopt_iter)
        xopt_iter = np.array([mixint.cast_to_mixed_integer( xopt_iter[i,:]) 
                        for i in range(xopt_iter.shape[0])]).reshape(xopt_iter.shape)
        xopt_iter = scaler.transform(xopt_iter)
        xopt_iter, _ = drop_duplicates(xopt_iter,np.zeros_like(xopt_iter))
        xopt_iter, _ = concat_to_existing(xnew,np.zeros_like(xnew), xopt_iter, np.zeros_like(xopt_iter))

        # run model at all candidate points
        start = time.time()
        yopt_iter = ParallelEvaluator(
          n_procs = n_procs).run(fun=fun,x=xopt_iter)
        
        lapse = np.round( ( time.time() - start )/60, 2)
        print(f'Check-optimal candidates: new {xopt_iter.shape[0]} simulations took {lapse} minutes')    

        # update the db of model evaluations, xdoe and ydoe
        xdoe_upd, ydoe_upd = concat_to_existing(xdoe,ydoe, xopt_iter,yopt_iter)
        xdoe_upd, ydoe_upd = drop_duplicates(xdoe_upd, ydoe_upd)
        
        # Drop yopt if it is not better than best design seen
        xopt = xdoe_upd[[np.argmin(ydoe_upd)],:]
        yopt = ydoe_upd[[np.argmin(ydoe_upd)],:]
        
        #if itr > 0:
        error = float(1 - yopt/yold)
        print(f'  rel_yopt_change = {error:.2E}')

        xdoe = np.copy(xdoe_upd)
        ydoe = np.copy(ydoe_upd)
        xold = np.copy(xopt)
        yold = np.copy(yopt)
        itr = itr+1

        lapse = np.round( ( time.time() - start_iter )/60, 2)
        print(f'Iteration {itr} took {lapse} minutes\n')

        if (np.abs(error) < tol):
            conv_iter += 1
            if (conv_iter >= min_conv_iter):
                print(f'Surrogate based optimization is converged.')
                break
        else:
            conv_iter = 0
    
    xopt = scaler.inverse_transform(xopt)
    
    # Re-Evaluate the last design to get all outputs
    outs = hpp_m.evaluate(*xopt[0,:])
    yopt = np.array(opt_sign*outs[[op_var_index]])[:,na]
    hpp_m.print_design(xopt[0,:], outs)

    n_model_evals = xdoe.shape[0] 
    
    lapse = np.round( ( time.time() - start_total )/60, 2)
    print(f'Optimization with {itr} iterations and {n_model_evals} model evaluations took {lapse} minutes\n')

    # Store results
    # -----------------
    design_df = pd.DataFrame(columns = list_vars, index=[name])
    for iv, var in enumerate(list_vars):
        design_df[var] = xopt[0,iv]
    for iv, var in enumerate(list_out_vars):
        design_df[var] = outs[iv]
    
    design_df['design obj'] = opt_var
    design_df['opt time [min]'] = lapse
    design_df['n_model_evals'] = n_model_evals
    
    design_df.T.to_csv(final_design_fn)
