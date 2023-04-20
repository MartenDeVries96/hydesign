# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:44:06 2023

@author: mikf
"""
import time
import numpy as np
from numpy import newaxis as na
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from smt.applications.ego import Evaluator
from smt.applications.mixed_integer import (
    MixedIntegerContext,
    FLOAT,
    INT,)
from smt.sampling_methods import LHS
from hydesign.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath
from hydesign.EGO_surrogate_based_optimization import (get_sm, eval_sm,
                                                       get_candiate_points, drop_duplicates, opt_sm,
                                                       concat_to_existing)
from sys import version_info
from openmdao.core.driver import Driver
import pickle
from datetime import datetime
import os


def surrogate_optimization(inputs): # Calling the optimization of the surrogate model
    x, kwargs = inputs
    mixint = MixedIntegerContext(kwargs['xtypes'], kwargs['xlimits'])
    return opt_sm(kwargs['sm'], mixint, x, fmin=kwargs['yopt'][0,0])

def surrogate_evaluation(inputs): # Evaluates the surrogate model
    seed, kwargs = inputs
    mixint = MixedIntegerContext(kwargs['xtypes'], kwargs['xlimits'])
    return eval_sm(
    kwargs['sm'], mixint, 
    scaler=kwargs['scaler'],
    seed=seed, #different seed on each iteration
    npred=kwargs['npred'],
    fmin=kwargs['yopt'][0,0],)

def model_evaluation(inputs): # Evaluates the model
    x, kwargs = inputs
    hpp_m = hpp_model(
            **kwargs,
            verbose=False)

    x = kwargs['scaler'].inverse_transform(x)
    return np.array(
        kwargs['opt_sign']*hpp_m.evaluate(*x[0,:])[kwargs['op_var_index']])


class ParallelEvaluator(Evaluator):
    """
    Implement Evaluator interface using multiprocessing Pool object (Python 3 only).
    """
    def __init__(self, n_procs = 31):
        self.n_procs = n_procs
        
    def run_ydoe(self, fun, x, **kwargs):
        n_procs = self.n_procs
        if version_info.major == 2:
            raise('version_info.major==2')
            
        with Pool(n_procs) as p:
            return np.array(p.map(fun, [(x[[i], :], kwargs) for i in range(x.shape[0])])).reshape(-1, 1)

    def run_both(self, fun, i, **kwargs):
        n_procs = self.n_procs
        if version_info.major == 2:
            raise('version_info.major==2')
            
        with Pool(n_procs) as p:
            return (p.map(fun, [((n + i * 100) * 100 + kwargs['n_seed'], kwargs) for n in np.arange(n_procs)]))
        
    def run_xopt_iter(self, fun, x, **kwargs):
        n_procs = self.n_procs
        if version_info.major == 2:
            raise('version_info.major==2')
            
        with Pool(n_procs) as p:
            return np.vstack(p.map(fun, [(x[[ii],:], kwargs) for ii in range(x.shape[0])]))
    
def derive_example_info(kwargs):
    example = kwargs['example']
    
    if example == None:
        kwargs['name'] = str(kwargs['name'])
        for x in ['longitude', 'latitude', 'altitude']:
            kwargs[x] = int(kwargs[x])
        kwargs['input_ts_fn'] = examples_filepath+str(kwargs['input_ts_fn'])
        kwargs['sim_pars_fn'] = examples_filepath+str(kwargs['sim_pars_fn'])
        
    else:
        examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
        
        try:
            ex_site = examples_sites.iloc[int(example),:]
    
            print('Selected example site:')
            print('---------------------------------------------------')
            print(ex_site.T)
    
            kwargs['name'] = ex_site['name']
            kwargs['longitude'] = ex_site['longitude']
            kwargs['latitude'] = ex_site['latitude']
            kwargs['altitude'] = ex_site['altitude']
            kwargs['input_ts_fn'] = examples_filepath+ex_site['input_ts_fn']
            kwargs['sim_pars_fn'] = examples_filepath+ex_site['sim_pars_fn']
            
        except:
            raise(f'Not a valid example: {int(example)}')
    
    return kwargs
           

def get_kwargs(kwargs):
    kwargs = derive_example_info(kwargs)
    for x in ['num_batteries', 'n_procs', 'n_doe', 'n_clusters',
              'n_seed', 'max_iter']:
        kwargs[x] = int(kwargs[x])
    
    if kwargs['final_design_fn'] == None:
        kwargs['final_design_fn'] = f'{kwargs["work_dir"]}design_hpp_{kwargs["name"]}_{kwargs["opt_var"]}.csv'  

    for x in ['opt_var', 'final_design_fn']:
        kwargs[x] = str(kwargs[x])
        
    return kwargs

class EfficientGlobalOptimizationDriver(Driver):
    def __init__(self, model, **kwargs):
        self.hpp_model = model
        self.kwargs = kwargs
        super().__init__(**kwargs)
    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        for k, v in self.kwargs.items():
            self.options.declare(k, v)
    def run(self, state=None):
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
        
        start_total = time.time()
        
        xtypes = kwargs['xtypes']
        xlimits = kwargs['xlimits']
        
        # Scale design variables
        scaler = MinMaxScaler()
        scaler.fit(xlimits.T)
        
      
        # START Parallel-EGO optimization
        # -------------------------------------------------------        
        
        # LHS intial doe
        mixint = MixedIntegerContext(xtypes, xlimits)
        sampling = mixint.build_sampling_method(
          LHS, criterion="maximin", random_state=kwargs['n_seed'])
        xdoe = sampling(kwargs['n_doe'])
        if state is not None:
            xdoe, _ = concat_to_existing(state, np.zeros_like(state), xdoe, np.zeros_like(xdoe))
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
        opt_var = kwargs['opt_var']
        opt_sign = -1
        if opt_var in list_minimize:
            opt_sign = 1
        
        kwargs['opt_sign'] = opt_sign
        kwargs['scaler'] = scaler
        kwargs['xtypes'] = xtypes
        kwargs['xlimits'] = xlimits
    
        hpp_m = self.hpp_model(**kwargs)
        
        print('\n\n')
        
        # Lists of all possible outputs, inputs to the hpp model
        # -------------------------------------------------------
        list_vars = hpp_m.list_vars
        list_out_vars = hpp_m.list_out_vars
        op_var_index = list_out_vars.index(opt_var)
        kwargs.update({'op_var_index': op_var_index})
        # Stablish types for design variables
        
        # Evaluate model at initial doe
        start = time.time()
        n_procs = kwargs['n_procs']
        PE = ParallelEvaluator(n_procs = n_procs)
        ydoe = PE.run_ydoe(fun=model_evaluation,x=xdoe, **kwargs)
        
        lapse = np.round((time.time() - start)/60, 2)
        print(f'Initial {xdoe.shape[0]} simulations took {lapse} minutes\n')
        
        # Initialize iterative optimization
        itr = 0
        error = 1e10
        conv_iter = 0
        yopt = ydoe[[np.argmin(ydoe)],:]
        kwargs['yopt'] = yopt
        yold = np.copy(yopt)
        # xold = None
        rec_vars = ['xopt_iter', 'yopt_iter', 'itr', 'start_iter', 'error']
        self.recorder = {r: [] for r in rec_vars}
        while itr < kwargs['max_iter']:
            self.recorder['itr'].append(itr)
            # Iteration
            start_iter = time.time()
            self.recorder['start_iter'].append('start_iter')
            # Train surrogate model
            np.random.seed(kwargs['n_seed'])
            sm = get_sm(xdoe, ydoe, mixint)
            kwargs['sm'] = sm
            
            # Evaluate surrogate model in a large number of design points
            # in parallel
            start = time.time()
            both = PE.run_both(surrogate_evaluation, itr, **kwargs)
            # with Pool(n_procs) as p:
            #     both = ( p.map(fun_par, (np.arange(n_procs)+itr*100) * 100 + itr) )
            xpred = np.vstack([both[ii][0] for ii in range(len(both))])
            ypred_LB = np.vstack([both[ii][1] for ii in range(len(both))])
            
            # Get candidate points from clustering all sm evalautions
            n_clusters = kwargs['n_clusters']
            xnew = get_candiate_points(
                xpred, ypred_LB, 
                n_clusters = n_clusters, 
                quantile = 1/(kwargs['npred']/n_clusters) ) 
                # request candidate points based on global evaluation of current surrogate 
                # returns best designs in n_cluster of points with outputs bellow a quantile
            lapse = np.round( ( time.time() - start )/60, 2)
            print(f'Update sm and extract candidate points took {lapse} minutes')
            
            # # optimize the sm starting on the cluster based candidates 
            xopt_iter = PE.run_xopt_iter(surrogate_optimization, xnew, **kwargs)
            xopt_iter = scaler.inverse_transform(xopt_iter)
            xopt_iter = np.array([mixint.cast_to_mixed_integer( xopt_iter[i,:]) 
                            for i in range(xopt_iter.shape[0])]).reshape(xopt_iter.shape)
            xopt_iter = scaler.transform(xopt_iter)
            xopt_iter, _ = drop_duplicates(xopt_iter,np.zeros_like(xopt_iter))
            xopt_iter, _ = concat_to_existing(xnew,np.zeros_like(xnew), xopt_iter, np.zeros_like(xopt_iter))
            self.recorder['xopt_iter'].append(xopt_iter)
            # run model at all candidate points
            start = time.time()
            yopt_iter = PE.run_ydoe(fun=model_evaluation,x=xopt_iter, **kwargs)
            self.recorder['yopt_iter'].append(yopt_iter)
            
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
            self.recorder['error'].append(error)
            print(f'  rel_yopt_change = {error:.2E}')
        
            xdoe = np.copy(xdoe_upd)
            ydoe = np.copy(ydoe_upd)
            # xold = np.copy(xopt)
            yold = np.copy(yopt)
            itr = itr+1
        
            lapse = np.round( ( time.time() - start_iter )/60, 2)
            print(f'Iteration {itr} took {lapse} minutes\n')
        
            if (np.abs(error) < kwargs['tol']):
                conv_iter += 1
                if (conv_iter >= kwargs['min_conv_iter']):
                    print('Surrogate based optimization is converged.')
                    break
            else:
                conv_iter = 0
        # refine = True
        # if refine:
        #     refine_input = kwargs.copy()
        #     # limits = 
        #     xopt_iter = PE.run_xopt_iter(surrogate_optimization, xnew, **kwargs)
        #     from scipy.optimize import minimize
        #     print('Current solution: ', yopt)
        #     def refine_solution(x):
        #         return kwargs['opt_sign']*hpp_m.evaluate(*kwargs['scaler'].inverse_transform(x.reshape(1, -1)).ravel())[kwargs['op_var_index']]
        #     res = minimize(refine_solution, xopt, method='SLSQP')
        #     xopt = res.x
        #     yopt = res.fun
        #     print('Refined solution: ', yopt)
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
        
        design_df.T.to_csv(kwargs['final_design_fn'])
        self.result = design_df
        self.recorder['meta_data'] = kwargs

class refine:
    def __init__(self, hpp_model, desvars, **kwargs):
        from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
        driver = ScipyOptimizeDriver(optimizer='SLSQP')
        self.hpp_m = hpp_model(**kwargs)
        prob = self.hpp_m.prob
        prob.driver = driver
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['includes'] = ['*']
        prob.driver.recording_options['record_inputs'] = True
        for k, v in desvars.items():
            prob.model.add_design_var(k, lower=v[1], upper=v[2])
        prob.model.add_objective(kwargs['opt_var'], scaler=kwargs['opt_sign'])
        prob.setup()   
        self.prob = prob
        
    def update_state(self, state):
        for k, v in state.items():
            try:
                c = self.prob[k]  # fail if k not exists
                v = np.array(v)
                if hasattr(c, 'shape') and c.shape != v.shape:
                    v = v.reshape(c.shape)
                self.prob[k] = v
            except KeyError:
                pass

    def evaluate(
        self,
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio
        ):
        return self.hpp_m.evaluate(clearance, sp, p_rated, Nwt, wind_MW_per_km2,
                solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
                b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio)
    
    def optimize(self, state, disp=True):
        self.prob.setup()
        self.update_state(state)
        t = time.time()
        self.prob.run_driver()
        self.prob.cleanup()
        if disp:
            print("Optimized in\t%.3fs" % (time.time() - t))



if __name__ == '__main__':
    inputs = {
        'example': 0,
        'name': None,
        'longitude': None,
        'latitude': None,
        'altitude': None,
        'input_ts_fn': None,
        'sim_pars_fn': None,

        'opt_var': "NPV_over_CAPEX",
        'rotor_diameter_m': 100,
        'hub_height_m': 120,
        'wt_rated_power_MW': 2,
        'surface_tilt_deg': 20,
        'surface_azimuth_deg': 180,
        'DC_AC_ratio': 1,
        'num_batteries': 1,
        'n_procs': 7,
        'n_doe': 50,
        'n_clusters': 10,
        'n_seed': 0,
        'max_iter': 10,
        'final_design_fn': 'hydesign_design_0.csv',
        'npred': 3e4,
        'tol': 1e-6,
        'min_conv_iter': 3,
        'work_dir': './',
        }

    kwargs = get_kwargs(inputs)
    kwargs['xtypes'] = [
        #clearance, sp, p_rated, Nwt, wind_MW_per_km2, 
        INT, INT, INT, INT, FLOAT, 
        #solar_MW, surface_tilt, surface_azimuth, DC_AC_ratio
        INT,FLOAT,FLOAT,FLOAT,
        #b_P, b_E_h , cost_of_battery_P_fluct_in_peak_price_ratio
        INT,INT,FLOAT]

    kwargs['xlimits'] = np.array([
        #clearance: min distance tip to ground
        [10, 60],
        #Specific Power
        [200, 400],
        #p_rated
        [1, 10],
        #Nwt
        [0, 400],
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
        [0, 100],
        #b_E_h in h
        [1, 10],
        #cost_of_battery_P_fluct_in_peak_price_ratio
        [0, 20],
        ])    
    EGOD = EfficientGlobalOptimizationDriver(model=hpp_model, **kwargs)
    # states that you want to include e.g. an edge case
    state = np.array([15, 292, 2, 145, 7, 3, 25, 178, 1.6, 29, 2, 4])
    kwargs['xlimits'] = state[:, na]*np.asarray([0.95, 1.05])
    if 1:
        EGOD.run(state=state)
        result = EGOD.result
        recorder = EGOD.recorder
        if not os.path.exists('Recordings'):
                       os.makedirs('Recordings')
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        with open(f'Recordings/rec_{date_time}.pkl', 'wb') as f:
            pickle.dump(recorder, f)
    else:
        with open('Recordings/rec_2023_04_13_10_21_23.pkl', 'rb') as f:
            recorder = pickle.load(f)
    kwargs = recorder['meta_data']
    
    hpp_list_vars = [   'clearance [m]', 
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
    init_state = np.asarray([[i, l, u] for i, l, u in zip([15, 292, 2, 145, 7, 3, 25, 178, 1.6, 29, 2, 4], kwargs['xlimits'][:,0], kwargs['xlimits'][:,1])])

    dic = {k: v for k, v in zip(hpp_list_vars, init_state)}

    def get_rotor_area(d): return np.pi*(d/2)**2
    def get_rotor_d(area): return 2*(area/np.pi)**0.5


    p_rated = dic['p_rated [MW]']
    sp = dic['sp [m2/W]']
    clearance = dic['clearance [m]']
    Nwt = dic['Nwt']
    wind_MW_per_km2 = dic['wind_MW_per_km2 [MW/km2]']
    b_E_h = dic['b_E_h [h]']
    b_P = dic['b_P [MW]']
    
    d = get_rotor_d(p_rated*1e6/sp)
    hh = (d/2)+clearance
    wind_MW = Nwt * p_rated
    Awpp = wind_MW / wind_MW_per_km2 
    #Awpp = Awpp + 1e-10*(Awpp==0)
    b_E = b_E_h * b_P

    state = {
        'hh': hh,
        'd': d,
        'p_rated': p_rated,
        'Nwt': Nwt,
        'Awpp': Awpp,
        'surface_tilt': dic['surface_tilt [deg]'],
        'surface_azimuth': dic['surface_azimuth [deg]'],
        'DC_AC_ratio': dic['DC_AC_ratio'], 
        'solar_MW': dic['solar_MW [MW]'],
        'b_P': dic['b_P [MW]'],
        'b_E': b_E,
        'cost_of_battery_P_fluct_in_peak_price_ratio': dic['cost_of_battery_P_fluct_in_peak_price_ratio']}
    REFINE = refine(hpp_model, desvars=state, **kwargs)
    REFINE.optimize({k: v[0] for k, v in state.items()})
    
