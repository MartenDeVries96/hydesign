import glob
import os

import time
import numpy as np
from numpy import newaxis as na
import pandas as pd
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import optimize
from scipy.stats import norm

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

from hpp_assembly import hpp_model, mkdir

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

def get_sm(xdoe, ydoe, mixint):
    '''
    Function that trains the surrogate and uses it to predict on random input points
    '''
    sm = KPLSK(
        corr="squar_exp",
        poly='linear',
        theta0=[1e0],
        theta_bounds=[1e-6, 1e3],
        n_start=20,
        print_global=False)

    sm.set_training_values(xdoe, ydoe)
    sm.train()

    return sm


def get_sm_pred(sm, mixint, seed=0, npred=1e3, fmin=1e10):
    '''
    Function that predicts sm on random input points
    '''

    npred = int(npred)
    np.random.seed(int(seed))
    sampling = mixint.build_sampling_method(Random)
    #sampling = mixint.build_sampling_method(
    #    LHS, criterion="maximin", random_state=int(seed))

    xpred = sampling(npred)    
    #ypred_LB = LCB(sm=sm, point=xpred)
    ypred_LB = EI(sm=sm, point=xpred, fmin=fmin)

    return xpred, ypred_LB

def opt_sm(sm, mixint, x0, fmin=1e10):
    '''
    Function that optimizes the surrogate based on lower confidence bound predictions
    '''

    ndims = mixint.get_unfolded_dimension()
    res = optimize.minimize(
        #fun = lambda x:  LCB(sm, x.reshape([1,ndims])),
        fun = lambda x:  EI(sm, x.reshape([1,ndims]), fmin=fmin),
        ## No jacobian for LCB. Only for actual sm evaluation
        #fun = lambda x:  KB(sm, x.reshape([1,ndims])),
        #jac = lambda x: np.stack([sm.predict_derivatives(
        #    x.reshape([1,ndims]), kx=i) 
        #    for i in range(ndims)] ).reshape([1,ndims]),
        x0 = x0.reshape([1,ndims]),
        method="SLSQP",
        bounds=mixint.get_unfolded_xlimits(),
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

def drop_duplicates(x,y):
    x_cols = [f'x{i}' for i in range(x.shape[1])]
    y_cols = [f'y{i}' for i in range(y.shape[1])]

    df = pd.DataFrame(
        data=x,
        columns=x_cols
        )
    for i,y_col in enumerate(y_cols):
        df[y_col] = y[:,i]

    df.drop_duplicates(subset=x_cols, inplace=True)

    return df.loc[:,x_cols].values, df.loc[:,y_cols].values

def concat_to_existing(x,y,xnew,ynew):
    x_concat, y_concat = drop_duplicates(
        np.vstack([x,xnew]),
        np.vstack([y,ynew])
        )
    return x_concat, y_concat

def print_design(x_opt, outs, list_vars, list_out_vars, xtypes):
    print() 
    print('Final design:') 

    for i_v, var in enumerate(list_vars):
        if xtypes[i_v]==INT:
            print(var+':', int(x_opt[i_v]))
        else:
            print(var+':', x_opt[i_v])
    print()    
    print()
    for i_v, var in enumerate(list_out_vars):
        print(f'{var}: {outs[i_v]:.3f}')
    print()




if __name__ == "__main__":


    from corres.auxiliar_functions import clean, mkdir
    from multiprocessing import Pool
    

    # Required inputs
    # -----------------
    longitude = 68.54220353096616
    latitude = 23.54209921071357
    altitude = 29.883557407411217 
    # altitude = None #if not known
    
    num_batteries = 2
    ems_type = 'cplex'
    opt_var = 'NPV_over_CAPEX'
    work_dir = './results_C/'
    n_procs = 3 # number of parallel process
    
    # paralel EGO parameters
    n_doe = 42
    n_seed = 0
    max_iter = 20
    tol = 1e-6
    min_conv_iter = 3
    
    # HPP model
    # -----------------
    print()
    print('Hydesign hybrid power plant sizing and design:')
    print('\n\n\n')
    print('Sizing a HPP plant at:')
    print()
    hpp_m = hpp_model(
            latitude,
            longitude,
            altitude,
            num_batteries = num_batteries,
            ems_type = ems_type,
            work_dir = work_dir,
            sim_pars_fn = 'hpp_pars.yml',
            #input_ts_fn = './results/input_ts.csv',#None, # If None then it computes the weather
            # -------------------------------
            input_ts_fn = None, # If None then it computes the weather
            price_fn = 'elec_price_t_new.csv', # If input_ts_fn is given it should include Price column.
            era5_zarr = '/groups/reanalyses/era5/app/era5.zarr', # location of wind speed renalysis
            ratio_gwa_era5 = '/groups/INP/era5/ratio_gwa2_era5.nc', # location of mean wind speed correction factor
            era5_ghi_zarr = '/groups/INP/era5/ghi.zarr', # location of GHI renalysis
            elevation_fn = '/groups/INP/era5/SRTMv3_plus_ViewFinder_coarsen.nc',
            genWT_fn='/home/jumu/Hydesign_openmdao_dev/hydesign/hydesign/Aug3/genWT_v3.nc',
            genWake_fn='/home/jumu/Hydesign_openmdao_dev/hydesign/hydesign/Aug3/genWake_v3.nc',
    )
    print('\n\n')
    
    # Lists of all possible outputs, inputs to the hpp model
    # -------------------------------------------------------
    list_out_vars = ['NPV_over_CAPEX',
                 'NPV [MEuro]',
                 'IRR',
                 'LCOE [Euro/MWh]',
                 'CAPEX [MEuro]',
                 'OPEX [MEuro]',
                 'penalty lifetime [MEuro]',
                 'GUF',
                 'grid [MW]',
                 'wind [MW]',
                 'solar [MW]',
                 #'surface_tilt [deg]',
                 #'surface_azimuth [deg]',
                 #'DC_AC_ratio',
                 'Battery Energy [MWh]',
                 'Battery Power [MW]',
                 'Total curtailment [GWh]',
                 'Awpp [km2]',
                 #'Apvp [km2]',
                 'Rotor diam [m]',
                 'Hub height [m]',
                 'Number_of_batteries',
                ]
    
    list_minimize = ['LCOE [Euro/MWh]']

    list_vars = ['clearance [m]', 
                 'sp [m2/W]', 
                 'p_rated [MW]', 
                 'Nwt', 
                 'wind_MW_per_km2 [MW/km2]', 
                 'solar_MW [MW]', 
                 'surface_tilt [deg]', 
                 'surface_azimuth [deg]', 
                 'DC_AC_ratio', 
                 'b_P [MW]', 
                 'b_E_h [h]',
                 'cost_of_battery_P_fluct_in_peak_price_ratio'
                ]
    
    # Get index of output var to optimize
    op_var_index = list_out_vars.index(opt_var)
    # Get sign to always write the optimization as minimize
    opt_sign = -1
    if opt_var in list_minimize:
        opt_sign = 1
    
    # Stablish types for design variables
    xtypes = [
        #clearance, sp, p_rated, Nwt, wind_MW_per_km2, 
        INT, INT, INT, INT, FLOAT, 
        #solar_MW, surface_tilt, surface_azimuth, DC_AC_ratio
        INT,FLOAT,FLOAT,FLOAT,
        #b_P, b_E_h , cost_of_battery_P_fluct_in_peak_price_ratio
        INT,INT,FLOAT]

    xlimits = np.array([
        #clearance: min distance tip to ground
        [10, 60],
        #Specific Power
        [200, 400],
        #p_rated
        [1, 10],
        #Nwt
        [0, 300],
        #wind_MW_per_km2
        [5, 9],
        #solar_MW
        [0, 400],
        #surface_tilt
        [0, 50],
        #surface_azimuth
        [150, 210],
        #DC_AC_ratio
        [1, 2.0],
        #b_P in MW
        [0, 300],
        #b_E_h in h
        [1, 10],
        #cost_of_battery_P_fluct_in_peak_price_ratio: limits battery deggradation in EMS optimization
        [0, 20],
        ])   
    
    # Create a parallel evaluator of the model
    # -------------------------------------------------------
    def fun(x): 
        try:
            return np.array(
                opt_sign*hpp_m.evaluate(*x[0,:])[op_var_index])
        except:
            print(f'x={x}')
    
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
    start_total = time.time()
    
    # Scale design variables
    scaler = MinMaxScaler()
    scaler.fit(xlimits.T)
    
    # LHS intial doe
    mixint = MixedIntegerContext(xtypes, xlimits)
    sampling = mixint.build_sampling_method(
       LHS, criterion="ese", random_state=n_seed)
    xdoe = sampling(n_doe)

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
    while itr < max_iter:
        # Iteration
        start_iter = time.time()

        # Train surrogate model
        sm = get_sm(xdoe, ydoe, mixint)
        
        # Evaluate surrogate model in a large number of design points
        # in parallel
        start = time.time()
        def fun_par(seed): return get_sm_pred(
            sm, mixint, 
            seed=seed, 
            npred=2e4,
            fmin=yopt[0,0],
        )
        with Pool(n_procs) as p:
            both = ( p.map(fun_par, np.arange(n_procs)+itr*100 ) )
        xpred = np.vstack([both[ii][0] for ii in range(len(both))])
        ypred_LB = np.vstack([both[ii][1] for ii in range(len(both))])
        
        # Get candidate points from clustering all sm evalautions
        xnew = get_candiate_points(
            xpred, ypred_LB, 
            n_clusters = n_procs, 
            quantile = 1e-4) 
            # request candidate points based on global evaluation of current surrogate 
            # returns best designs in n_cluster of points with outputs bellow a quantile
        lapse = np.round( ( time.time() - start )/60, 2)
        print(f'Update sm and extract candidate points took {lapse} minutes')

        # run model on candidates
        start = time.time()
        ynew = ParallelEvaluator(
            n_procs = n_procs).run(fun=fun,x=xnew)
        lapse = np.round( ( time.time() - start )/60, 2)
        print(f'New {xnew.shape[0]} simulations took {lapse} minutes')
        
        # optimize the sm starting on the cluster based candidates 
        def fun_opt(x): return opt_sm(sm, mixint, x, fmin=yopt[0,0])
        with Pool(n_procs) as p:
            xopt = np.vstack(
                    p.map(fun_opt, [xnew[ii,:] 
                    for ii in range(xnew.shape[0])] ) 
                )
        xopt = smt.applications.mixed_integer.cast_to_discrete_values(
            xtypes, xopt)
        xopt, _ = drop_duplicates(xopt,np.zeros_like(xopt))

        # run model at all surrogate optimal points
        start = time.time()
        yopt = ParallelEvaluator(
          n_procs = n_procs).run(fun=fun,x=xopt)
        
        lapse = np.round( ( time.time() - start )/60, 2)
        print(f'Check-optimal candidates: new {xopt.shape[0]} simulations took {lapse} minutes')    

        # update the db of model evaluations, xdoe and ydoe
        xnew, ynew = concat_to_existing(xnew,ynew, xopt,yopt)
        xdoe_upd, ydoe_upd = concat_to_existing(xdoe,ydoe, xnew,ynew)
          
        # Drop yopt if it is not better than best design seen
        if np.min(yopt) > np.min(ydoe):
            xopt = xdoe[[np.argmin(ydoe)],:]
            yopt = ydoe[[np.argmin(ydoe)],:]
        else:
            xopt = xopt[[np.argmin(yopt)],:]
            yopt = yopt[[np.argmin(yopt)],:]

        if itr > 0:
            error = np.linalg.norm(
               scaler.transform( np.atleast_2d(np.array(xopt)) ) +
               - scaler.transform( np.atleast_2d(np.array(xold)) )
            )

        xdoe = xdoe_upd
        ydoe = ydoe_upd
        xold = xopt
        yold = yopt
        itr = itr+1

        lapse = np.round( ( time.time() - start_iter )/60, 2)
        print(f'Iteration {itr} took {lapse} minutes\n')

        if (error < tol):
            conv_iter += 1
            #print(f'     |delta_xopt| = {error:.2E} < {tol:.4E}\n')
            if (conv_iter >= min_conv_iter):
                print(f'Surrogate based optimization is converged.')
                print(f'     |delta_xopt| = {error:.2E}\n') 
                break
        else:
            conv_iter = 0
            #print(f'     |delta_xopt| = {error:.2E}\n') 

    # Re-Evaluate the last design to get all outputs
    outs = hpp_m.evaluate(*xopt[0,:])
    yopt = np.array(opt_sign*outs[[op_var_index]])[:,na]
    print_design(xopt[0,:], outs, list_vars, list_out_vars, xtypes)

    lapse = np.round( ( time.time() - start_total )/60, 2)
    print(f'Optimization with {itr} iterations took {lapse} minutes\n')

    # Store results
    # -----------------
    design_df = pd.DataFrame()
    for iv, var in enumerate(list_vars):
        design_df[var] = xopt[0,iv]
    for iv, var in enumerate(list_out_vars):
        design_df[var] = outs[iv]

    design_df['opt time [min]'] = lapse

    design_df.to_csv(f'{work_dir}design_df.csv')

