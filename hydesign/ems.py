# %%

import glob
import os
import time
import copy

# basic libraries
import numpy as np
from numpy import newaxis as na
import pandas as pd
import openmdao.api as om
import yaml

import xarray as xr
from docplex.mp.model import Model

class ems(om.ExplicitComponent):
    """Energy management optimization model"""

    def __init__(
        self, 
        N_time, 
        life_h = 25*365*24, 
        ems_type='cplex'):

        super().__init__()
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.life_h = int(life_h)

    def setup(self):
        self.add_input(
            'wind_t',
            desc="WPP power time series",
            units='MW',
            shape=[self.N_time])
        self.add_input(
            'solar_t',
            desc="PVP power time series",
            units='MW',
            shape=[self.N_time])
        self.add_input(
            'price_t',
            desc="Electricity price time series",
            shape=[self.N_time])
        self.add_input(
            'b_P',
            desc="Battery power capacity",
            units='MW')
        self.add_input(
            'b_E',
            desc="Battery energy storage capacity")
        self.add_input(
            'G_MW',
            desc="Grid capacity",
            units='MW')
        self.add_input(
            'battery_depth_of_discharge',
            desc="battery depth of discharge",
            units='MW')
        self.add_input(
            'battery_charge_efficiency',
            desc="battery charge efficiency")
        self.add_input(
            'peak_hr_quantile',
            desc="Quantile of price tim sereis to define peak price hours (above this quantile).\n"+
                 "Only used for peak production penalty and for cost of battery degradation.")
        self.add_input(
            'cost_of_battery_P_fluct_in_peak_price_ratio',
            desc="cost of battery power fluctuations computed as a peak price ratio.")
        self.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")

        # ----------------------------------------------------------------------------------------------------------
        self.add_output(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'price_t_ext',
            desc="Electricity price time series",
            shape=[self.life_h])

        self.add_output(
            'hpp_t',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_t',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])
        self.add_output(
            'penalty_t',
            desc="penalty for not reaching expected energy productin at peak hours",
            shape=[self.life_h])        

    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):
        
        wind_t = inputs['wind_t']
        solar_t = inputs['solar_t']
        price_t = inputs['price_t']

        b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']
    
        if self.ems_type == 'cplex':
            ems_WSB = ems_cplex
        elif self.ems_type == 'pyomo':
            ems_WSB = ems_Wind_Solar_Battery_Pyomo
        else:
            ems_WSB = ems_rule_based
        
        # Avoid running an expensive optimization based ems if there is no battery
        if ( b_P <= 1e-2 ) or (b_E == 0):
            ems_WSB = ems_rule_based
    
        battery_depth_of_discharge = inputs['battery_depth_of_discharge']
        battery_charge_efficiency = inputs['battery_charge_efficiency']
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        cost_of_battery_P_fluct_in_peak_price_ratio = inputs['cost_of_battery_P_fluct_in_peak_price_ratio'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs[
            'n_full_power_hours_expected_per_day_at_peak_price'][0]
        
        # Build a sintetic time to avoid problems with time sereis 
        # indexing in ems
        WSPr_df = pd.DataFrame(
            index=pd.date_range(
                start='01-01-1990 00:00',
                periods=len(wind_t),
                freq='1h'))

        WSPr_df['wind_t'] = wind_t
        WSPr_df['solar_t'] = solar_t
        WSPr_df['price_t'] = price_t
        WSPr_df['E_batt_MWh_t'] = b_E[0]
        
        #print(WSPr_df.head())

        P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts = ems_WSB(
            wind_ts = WSPr_df.wind_t,
            solar_ts = WSPr_df.solar_t,
            price_ts = WSPr_df.price_t,
            P_batt_MW = b_P[0],
            E_batt_MWh_t = WSPr_df.E_batt_MWh_t,
            hpp_grid_connection = G_MW[0],
            battery_depth_of_discharge = battery_depth_of_discharge[0],
            charge_efficiency = battery_charge_efficiency[0],
            peak_hr_quantile = peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
        )

        # Extend (by repeating them and stacking) all variable to full lifetime 
        outputs['wind_t_ext'] = expand_to_lifetime(
            wind_t, life_h = self.life_h)
        outputs['solar_t_ext'] = expand_to_lifetime(
            solar_t, life_h = self.life_h)
        outputs['price_t_ext'] = expand_to_lifetime(
            price_t, life_h = self.life_h)
        outputs['hpp_t'] = expand_to_lifetime(
            P_HPP_ts, life_h = self.life_h)
        outputs['hpp_curt_t'] = expand_to_lifetime(
            P_curtailment_ts, life_h = self.life_h)
        outputs['b_t'] = expand_to_lifetime(
            P_charge_discharge_ts, life_h = self.life_h)
        outputs['b_E_SOC_t'] = expand_to_lifetime(
            E_SOC_ts[:-1], life_h = self.life_h + 1)
        outputs['penalty_t'] = expand_to_lifetime(
            penalty_ts, life_h = self.life_h)

class ems_long_term_operation(om.ExplicitComponent):
    """Energy management model"""

    def __init__(
        self, 
        N_time,
        num_batteries = 1,
        n_steps_in_LoH = 30,
        life_h = 25*365*24, 
        ):

        super().__init__()
        self.N_time = N_time
        self.life_h = life_h
        self.num_batteries = num_batteries
        self.n_steps_in_LoH = n_steps_in_LoH

    def setup(self):
        self.add_input(
            'ii_time',
            desc="indices on the liftime timeseries."+
                " Hydesign operates in each range at constant battery health",
            shape=[self.n_steps_in_LoH*self.num_batteries + 1 ])
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.n_steps_in_LoH*self.num_batteries + 1])
        self.add_input(
            'SoH_pv',
            desc="PV state of health time series",
            shape=[self.life_h]) 
        self.add_input(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'price_t_ext',
            desc="Electricity price time series",
            shape=[self.life_h])
        self.add_input(
            'b_P',
            desc="Battery power capacity",
            units='MW')
        self.add_input(
            'b_E',
            desc="Battery energy storage capacity")
        self.add_input(
            'G_MW',
            desc="Grid capacity",
            units='MW')
        self.add_input(
            'battery_depth_of_discharge',
            desc="battery depth of discharge",
            units='MW')
        self.add_input(
            'battery_charge_efficiency',
            desc="battery charge efficiency")
        self.add_input(
            'hpp_curt_t',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'b_t',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])
        self.add_input(
            'peak_hr_quantile',
            desc="Quantile of price tim sereis to define peak price hours (above this quantile).\n"+
                 "Only used for peak production penalty and for cost of battery degradation.")
        self.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")
        
        # -------------------------------------------------------

        self.add_output(
            'hpp_t_with_deg',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t_with_deg',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_t_with_deg',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_E_SOC_t_with_deg',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])
        self.add_output(
            'penalty_t_with_deg',
            desc="penalty for not reaching expected energy productin at peak hours",
            shape=[self.life_h])   
        self.add_output(
            'total_curtailment',
            desc="total curtailment in the lifetime",
            units='GW*h',
           )
        

    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):
        
        ii_time = inputs['ii_time']
        SoH = inputs['SoH']
        SoH_pv = inputs['SoH_pv']
        wind_t_ext = inputs['wind_t_ext']
        solar_t_ext = inputs['solar_t_ext']
        price_t_ext = inputs['price_t_ext']
        b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']
        battery_depth_of_discharge = inputs['battery_depth_of_discharge']
        battery_charge_efficiency = inputs['battery_charge_efficiency']
        hpp_curt_t = inputs['hpp_curt_t']
        b_t = inputs['b_t']
        b_E_SOC_t = inputs['b_E_SOC_t']
        
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs['n_full_power_hours_expected_per_day_at_peak_price'][0]

        life_h = self.life_h
        
        # Operate in time intervals
        
        time_intervals = []
        for ii in range(len(ii_time)-1):
            time_intervals += [[int(ii_time[ii]), int(ii_time[ii+1])]]
        time_intervals += [[int(ii_time[ii+1]), life_h-1]]

        Hpp_deg = np.zeros_like(wind_t_ext) 
        P_curt_deg = np.zeros_like(wind_t_ext)
        b_t_sat = np.zeros_like(wind_t_ext) 
        b_E_SOC_t_sat = np.zeros_like(b_E_SOC_t)
        penalty_t_with_deg_all = np.zeros_like(wind_t_ext)

        for ii, times_int in enumerate(time_intervals):
            times_op = slice(*times_int,1)

            if ii==0:
                b_E_SOC_0 = None
            else:
                b_E_SOC_0 = b_E_SOC_t_sat_aux[-1]

            Hpp_deg_aux, P_curt_deg_aux, b_t_sat_aux, \
            b_E_SOC_t_sat_aux, penalty_t_with_deg = operation_solar_batt_deg(
                pv_degradation = SoH_pv[times_int[0]],
                batt_degradation = SoH[ii],
                wind_t = wind_t_ext[times_op],
                solar_t = solar_t_ext[times_op],
                hpp_curt_t = hpp_curt_t[times_op],
                b_t = b_t[times_op],
                b_E_SOC_t = b_E_SOC_t[times_op],
                G_MW = G_MW[0],
                b_E = b_E[0],
                battery_depth_of_discharge = battery_depth_of_discharge[0],
                battery_charge_efficiency = battery_charge_efficiency[0],
                b_E_SOC_0 = b_E_SOC_0,
                price_ts = price_t_ext[times_op],
                peak_hr_quantile = peak_hr_quantile,
                n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
            )

            Hpp_deg[times_op] = Hpp_deg_aux
            P_curt_deg[times_op] = P_curt_deg_aux
            b_t_sat[times_op] = b_t_sat_aux
            b_E_SOC_t_sat[times_op] =  b_E_SOC_t_sat_aux[:-1]
            penalty_t_with_deg_all[times_op] = penalty_t_with_deg
            
        b_E_SOC_t_sat[-1] = b_E_SOC_t_sat[0]
            
        outputs['hpp_t_with_deg'] = Hpp_deg
        outputs['hpp_curt_t_with_deg'] = P_curt_deg
        outputs['b_t_with_deg'] = b_t_sat
        outputs['b_E_SOC_t_with_deg'] = b_E_SOC_t_sat
        outputs['penalty_t_with_deg'] = penalty_t_with_deg_all
        outputs['total_curtailment'] = P_curt_deg.sum()



# -----------------------------------------------------------------------
# Auxiliar functions for ems modelling
# -----------------------------------------------------------------------

def expand_to_lifetime(x, life_h = 25*365*24):

    len_x = len(x)
    N_repeats = int(np.ceil(life_h/len_x))

    return np.tile(x,N_repeats)[:life_h]
    

def ems_cplex(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,    
    batch_size = 2*24,
):
    
    # split in batches, ussually a week
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2]+batches_all[-1]
    
    # allocate vars
    P_HPP_ts = np.array([]) 
    P_curtailment_ts = np.array([])
    P_charge_discharge_ts = np.array([]) 
    E_SOC_ts = np.array([])
    penalty_ts = np.array([])
    
    #print('\n\nEMS solved with pyomo\n')
    for ib, batch in enumerate(batches):
        wind_ts_sel = wind_ts.iloc[batch]
        solar_ts_sel = solar_ts.iloc[batch]
        price_ts_sel = price_ts.iloc[batch]
        E_batt_MWh_t_sel = E_batt_MWh_t.iloc[batch]
        
        #print(f'batch {ib+1} out of {len(batches)}')
        P_HPP_ts_batch, P_curtailment_ts_batch, P_charge_discharge_ts_batch,\
        E_SOC_ts_batch, penalty_batch = ems_cplex_parts(
            wind_ts = wind_ts_sel,
            solar_ts = solar_ts_sel,
            price_ts = price_ts_sel,
            P_batt_MW = P_batt_MW,
            E_batt_MWh_t = E_batt_MWh_t_sel,
            hpp_grid_connection = hpp_grid_connection,
            battery_depth_of_discharge = battery_depth_of_discharge,
            charge_efficiency = charge_efficiency,
            peak_hr_quantile = peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
        )
        
        P_HPP_ts = np.append(P_HPP_ts, P_HPP_ts_batch)
        P_curtailment_ts = np.append(
            P_curtailment_ts, P_curtailment_ts_batch)
        P_charge_discharge_ts = np.append(
            P_charge_discharge_ts,P_charge_discharge_ts_batch)
        E_SOC_ts = np.append(E_SOC_ts, E_SOC_ts_batch)
        penalty_ts = np.append(penalty_ts, penalty_batch)
        
    E_SOC_ts = np.append(E_SOC_ts, E_SOC_ts[0])
    
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts


def ems_cplex_parts(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,
):
    """EMS solver implemented in cplex"""
    
    # Penalties 
    N_t = len(price_ts.index) 
    N_days = N_t/24
    e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price*hpp_grid_connection 
    e_peak_period_expected = e_peak_day_expected*N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts>=price_peak)[0]
    
    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max<0] = 0
    price_ts_to_max.iloc[:-1] = 0.5*price_ts_to_max.iloc[:-1].values + 0.5*price_ts_to_max.iloc[1:].values
        
    mdl = Model(name='EMS')
    mdl.context.cplex_parameters.threads = 1
    # CPLEX parameter pg 87 Emphasize feasibility over optimality
    # mdl.context.cplex_parameters.emphasis.mip = 1 
    #mdl.context.cplex_parameters.timelimit = 1e-2
    #mdl.context.cplex_parameters.mip.limits.strongit = 3
    #mdl.context.cplex_parameters.mip.strategy.search = 1 #  branch and cut strategy; disable dynamic
    
    #cpx = mdl.get_cplex()
    # cpx.parameters.mip.tolerances.integrality.set(0)
    # cpx.parameters.simplex.tolerances.markowitz.set(0.999)
    # cpx.parameters.simplex.tolerances.optimality.set(1e-6)#1e-9)
    # cpx.parameters.simplex.tolerances.feasibility.set(1e-5)#1e-9)
    # cpx.parameters.mip.pool.intensity.set(2)
    # cpx.parameters.mip.pool.absgap.set(1e75)
    # cpx.parameters.mip.pool.relgap.set(1e75)
    # cpx.parameters.mip.limits.populate.set(50)    
    
    time = price_ts.index
    # time set with an additional time slot for the last soc
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

    # Variables definition
    P_HPP_t = mdl.continuous_var_dict(
        time, lb=0, ub=hpp_grid_connection, 
        name='HPP power output')
    P_curtailment_t = mdl.continuous_var_dict(
        time, lb=0, 
        name='Curtailment')

    # Power charge/discharge from battery
    # Lower bound as large negative number in order to allow the variable to
    # have either positive or negative values
    P_charge_discharge = mdl.continuous_var_dict(
        time, lb=-P_batt_MW/charge_efficiency, ub=P_batt_MW*charge_efficiency, 
        name='Battery power')
    # Battery energy level, energy stored
    E_SOC_t = mdl.continuous_var_dict(
        SOCtime, lb=0, #ub=E_batt_MWh_t.max(), 
        name='Energy level')
    
    penalty = mdl.continuous_var(name='penalty', lb=-1e12)
    e_penalty = mdl.continuous_var(name='e_penalty', lb=-1e12)
    
    # Piecewise function for "absolute value" function
    fabs = mdl.piecewise(-1, [(0,0)], 1)
    
    mdl.maximize(
        # revenues and OPEX
        mdl.sum(
            price_ts[t] * P_HPP_t[t]
            for t in time) - penalty \
        # Add cost for rapid charge-discharge for limiting the battery life use
        - mdl.sum(
           fabs(P_charge_discharge[t + pd.Timedelta('1hour')] - \
                P_charge_discharge[t])*cost_of_battery_P_fluct_in_peak_price_ratio*price_ts_to_max[t]
           for t in time[:-1])  
        #- mdl.sum(
        #    fabs(P_charge_discharge[t + pd.Timedelta('1hour')] - \
        #         P_charge_discharge[t])*cost_of_battery_P_fluct_in_peak_price_ratio*price_peak
        #    for t in time[:-1])  
    ) 
        
    #Constraints
    mdl.add_constraint(
       e_penalty == ( e_peak_period_expected - mdl.sum(P_HPP_t[time[i]] for i in peak_hours_index) ) 
       )
    # Piecewise function for "only positive" function
    f1 = mdl.piecewise(0, [(0,0)], 1)
    mdl.add_constraint( penalty == price_peak*f1(e_penalty) )
        
    # Intitial and end SOC
    mdl.add_constraint( E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]] )
    
    # SOC at the end of the year has to be equal to SOC at the beginning of the year
    mdl.add_constraint( E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]] )

    # pircewise linear representation of charge vs dischrage effciency 
    f2 = mdl.piecewise(charge_efficiency,[(0,0)],1/charge_efficiency)
    
    for t in time:
        # Time index for successive time step
        tt = t + pd.Timedelta('1hour')
        # Delta_t of 1 hour
        dt = 1
        
        # Only one variable for battery
        mdl.add_constraint(
            P_HPP_t[t] == wind_ts[t] +
            solar_ts[t] +
            - P_curtailment_t[t] +
            P_charge_discharge[t])
        
        # charge/dischrage equation
        mdl.add_constraint(
            E_SOC_t[tt] == E_SOC_t[t] - 
            f2(P_charge_discharge[t]) * dt)
        
        # Constraining battery energy level to minimum battery level
        mdl.add_constraint(
            E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t]
        )
        
        # Constraining battery energy level to maximum battery level
        mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])

        # Battery charge/discharge within its power rating
        mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW*charge_efficiency)
        mdl.add_constraint(P_charge_discharge[t] >= -P_batt_MW/charge_efficiency)
        

    # Solving the problem
    sol = mdl.solve(
        log_output=False)
        #log_output=True)

    
    #print(mdl.export_to_string())
    #sol.display() 
    
    P_HPP_ts = pd.DataFrame.from_dict(
        sol.get_value_dict(P_HPP_t), orient='index').loc[:,0].values

    P_curtailment_ts = pd.DataFrame.from_dict(
        sol.get_value_dict(P_curtailment_t), orient='index').loc[:,0].values

    P_charge_discharge_ts = pd.DataFrame.from_dict(
        sol.get_value_dict(P_charge_discharge), orient='index').loc[:,0].values

    E_SOC_ts = pd.DataFrame.from_dict(
        sol.get_value_dict(E_SOC_t), orient='index').loc[:,0].values
    
    #make a time series like P_HPP with a constant penalty 
    penalty_2 = sol.get_value(penalty)
    penalty_ts = np.ones(N_t) * (penalty_2/N_t)
    
    mdl.end()
    
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts

def ems_rule_based(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5,
    n_full_power_hours_expected_per_day_at_peak_price = 3,
):
    G_MW = hpp_grid_connection
    
    td_date = wind_ts.index[1] - wind_ts.index[0]
    dt = td_date.total_seconds()/3600
    
    if P_batt_MW == 0:
        n_bat_h = 0
    else:
        n_bat_h = int(np.ceil(np.max(E_batt_MWh_t/P_batt_MW) ))
    h_charge =  list(np.arange(np.maximum(7-n_bat_h,0),7)) + \
                list(np.arange(np.maximum(18-n_bat_h,10),18))
    
    # Basic values
    H0 = wind_ts + solar_ts
    H_no_excs = np.minimum(H0.values, hpp_grid_connection)
    H_excs = np.maximum(H0.values - hpp_grid_connection, 0)
    H_charge = wind_ts.index.hour.isin(h_charge) * H_no_excs
    H_excs = np.minimum( H_excs+H_charge, H0)
    H_excs_prev = np.hstack([[0], H_excs[:-1]])   

    H_excs_battery_limit = np.minimum(H_excs, P_batt_MW)
    E_excs = np.cumsum(H_excs_battery_limit)*dt*charge_efficiency

    P_to_G = np.maximum(hpp_grid_connection - H0, 0)
    P_max_poss_by_bat = np.minimum(P_to_G, P_batt_MW)

    # identify time indices of battery operation
    it = np.where( (H_excs > 0) & (H_excs_prev == 0) )[0]
    ind = np.where(np.diff(it,prepend=0)>1)[0]  
    idt = np.array( sorted(it[ind]) ) - 1

    it_dp = np.where( (H_excs == 0) & (H_excs_prev > 0) )[0]
    ind_dp = np.where(np.diff(it_dp,prepend=0)>1)[0]  
    idt_dp = np.array( sorted(it_dp[ind_dp]) ) 
    
    # print()
    # print()
    # print()
    # print('before:')
    # print('idt:',idt)
    # print('idt_dp:',idt_dp)
    
    if len(idt) == 0:
        idt = np.array([0])
        idt_dp = np.array([len(H_excs)])
        
    if len(idt_dp) == 0:
        idt_dp = np.array([len(H_excs)])
        
    if idt[0] > idt_dp[0]:
        idt = np.hstack([0, idt])
        
    if len(idt) > len(idt_dp):
        idt_dp = np.hstack([idt_dp, len(H_excs)])
        
    # print()
    # print('after:')
    # print('idt:',idt)
    # print('idt_dp:',idt_dp)

    E_actual_without_min_dep = 0*H0.values
    E_excs_old = copy.copy(E_excs)
    for ii in range(len(idt)):
        # Charge
        i_start_ch = idt[ii]
        i_end_ch = idt_dp[ii]
        E_actual_without_min_dep[i_start_ch:i_end_ch+1] = np.minimum(
            E_excs_old[i_start_ch:i_end_ch+1],
            E_batt_MWh_t.values[i_start_ch:i_end_ch+1])

        if ii == len(idt) -1:
            ii_next = len(E_excs_old)-1
        else:
            ii_next = idt[ii+1]
        E_excs_old = np.maximum(E_excs_old - E_excs_old[ii_next], 0)

        # Discharge
        i_start_dsch = idt_dp[ii]
        if ii == len(idt) -1:
            i_end_dsch = len(E_excs_old)
        else:
            i_end_dsch = idt[ii+1]

        for jj in range(i_start_dsch+1,i_end_dsch):
            E_actual_without_min_dep[jj] = np.maximum(E_actual_without_min_dep[jj-1] - P_max_poss_by_bat[jj-1]*dt,0)

            
    E_actual = np.maximum(E_actual_without_min_dep, (1-battery_depth_of_discharge)*E_batt_MWh_t.values)
    E_actual = np.minimum(E_actual, E_batt_MWh_t.values)
        
    P_actual_battery_discharge = -np.diff(E_actual, append=0)/dt        
    P_actual_battery_discharge = np.maximum(P_actual_battery_discharge,-P_batt_MW)
    P_actual_battery_discharge = np.minimum(P_actual_battery_discharge,P_batt_MW)
    
    P_actual_battery_discharge = np.maximum(H0.values + P_actual_battery_discharge,0) - H0.values
    
    H_actual = np.minimum( H0.values + P_actual_battery_discharge, hpp_grid_connection)
    P_curt = np.maximum( H0.values + P_actual_battery_discharge - hpp_grid_connection, 0)

    P_HPP_ts = pd.Series(index=H0.index, data=H_actual)
    P_curtailment_ts = pd.Series(index=H0.index, data=P_curt)

    mask = P_actual_battery_discharge < 0

    P_charge_discharge_ts = pd.DataFrame(index=H0.index, data=P_actual_battery_discharge)

    time = H0.index
    # time set with an additional time slot for the last soc
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))
    E_SOC_ts = pd.DataFrame(
        index=SOCtime, 
        data=np.append(E_actual,E_actual[-1])
    )
    
    N_t = len(price_ts.values) 
    N_days = N_t/24
    e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price*G_MW 
    e_peak_period_expected = e_peak_day_expected*N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts>=price_peak)[0]
    e_penalty = e_peak_period_expected - np.sum([P_HPP_ts.values[i] for i in peak_hours_index])
    penalty = price_peak*np.maximum(0, e_penalty)
    penalty_ts = np.ones(N_t) * (penalty/N_t)

    return P_HPP_ts.values, \
        P_curtailment_ts.values, \
        P_charge_discharge_ts.loc[:,0].values, \
        E_SOC_ts.loc[:,0].values, \
        penalty_ts 
    
def ems_Wind_Solar_Battery_Pyomo_parts(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,
):
    """
    Method to calculate sizing of wind and solar and battery

    Returns
    -------
    Capacity of Wind Power
    Capacity of Solar Power
    Capacity of Batteru
    HPP power output timeseries
    HPP power curtailment timeseries
    HPP total CAPEX
    HPP total OPEX
    Levelised cost of energy
    """
    
    import pyomo.environ as pyo

    # extract parameters into the variable space
    #globals().update(self.__dict__)

    # Penalties 
    N_t = len(price_ts.index) 
    N_days = N_t/24
    e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price*hpp_grid_connection 
    e_peak_period_expected = e_peak_day_expected*N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts>=price_peak)[0]

    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max<0] = 0
    price_ts_to_max.iloc[:-1] = 0.5*price_ts_to_max.iloc[:-1].values + 0.5*price_ts_to_max.iloc[1:].values

    time = price_ts.index

    # time set with an additional time slot for the last soc
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))


    model = pyo.ConcreteModel()

    ## Variables ##
    model.IDX1 = range(len(time))
    model.IDX2 = range(1)
    model.IDX3 = [t for t in range(len(time)) if t in peak_hours_index]
    model.IDX4 = range(len(SOCtime))
    model.IDX5 = range(len(time)-1)

    model.P_HPP_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals, bounds=(0,hpp_grid_connection))
    model.P_curtailment_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals)

    # Power charge/discharge from battery with Lower/Upper bounds
    model.P_charge_discharge_t = pyo.Var(
        model.IDX1, 
        domain=pyo.Reals, 
        bounds=(-2*P_batt_MW, 2*P_batt_MW) # excess bounds
    )
    # Battery energy level
    model.E_SOC_t = pyo.Var(model.IDX4, domain=pyo.NonNegativeReals, bounds=(0,1e12)) #bounds=(0,E_batt_MWh))


    ## Constraints ##
    model.curtailment_constraint = pyo.ConstraintList()
    model.power_constraint = pyo.ConstraintList()
    model.charge_discharge_constraint = pyo.ConstraintList()
    model.battery_energy_constraint = pyo.ConstraintList()
    model.battery_energy_min_constraint = pyo.ConstraintList()
    model.battery_dynamics_constraint = pyo.ConstraintList()
    model.penalty_constraint = pyo.ConstraintList()

    model.SOC_initial_condition = pyo.Constraint(
        expr = model.E_SOC_t[0] == 0.5 * E_batt_MWh_t[0])

    # SOC at the end of the year has to be equal to SOC at the beginning of the year
    model.SOC_final = pyo.Constraint(
        expr = model.E_SOC_t[len(time) - 1] == 0.5 * E_batt_MWh_t[len(time) - 1])

    # x-values of the piece-wise function
    f_piecewise_x_vals = [-P_batt_MW, 0, P_batt_MW]
    # y-values of the piece-wise function
    f_piecewise_y_vals =  [-P_batt_MW/charge_efficiency, 0, P_batt_MW*charge_efficiency]

    model.P_charge_discharge_with_eff_t = pyo.Var(
        model.IDX1, 
        domain=pyo.Reals, 
        bounds=(-P_batt_MW/charge_efficiency, P_batt_MW*charge_efficiency)
    )
    model.battery_eff_constraint = pyo.Piecewise(
            model.IDX1,
            model.P_charge_discharge_with_eff_t, 
            model.P_charge_discharge_t, 
            pw_pts=f_piecewise_x_vals, 
            f_rule=f_piecewise_y_vals, 
            pw_constr_type='EQ', 
            #pw_repn='SOS2',
            #pw_repn='CC',
            #pw_repn='DCC',   
            pw_repn='DLOG',
            force_pw=False,
            warn_domain_coverage=False,
            unbounded_domain_var=True
            )


    model.penalty = pyo.Var(model.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))
    model.e_penalty = pyo.Var(model.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))


    model.penalty_constraint.add(
       model.e_penalty[0] == ( e_peak_period_expected - sum(
           model.P_HPP_t[t] for t in model.IDX3) ) 
       )

    model.penalty_constraint_pw = pyo.Piecewise(
            model.IDX2,
            model.penalty, 
            model.e_penalty, 
            pw_pts=[-1e12,0,1e12], 
            f_rule=[0,0,price_peak*1e12], 
            pw_constr_type='EQ', 
            #pw_repn='SOS2',
            #pw_repn='CC',
            #pw_repn='DCC',   
            pw_repn='DLOG',
            force_pw=False,
            warn_domain_coverage=False,
            #unbounded_domain_var=True
            )


    # Delta_t of 1 hour
    dt = 1

    for t in range(0,len(time)):

        # Constraining battery energy level to maximum battery level
        model.battery_energy_constraint.add(
            model.E_SOC_t[t] <= E_batt_MWh_t[t])
        # Constraining battery energy level to minimum battery level
        model.battery_energy_min_constraint.add(
            model.E_SOC_t[t] >= \
                (1 - battery_depth_of_discharge) * E_batt_MWh_t[t])
        # print(battery_depth_of_discharge)

        #Power constraint
        model.power_constraint.add(
           model.P_HPP_t[t] == wind_ts[t] +\
               solar_ts[t]  -\
               model.P_curtailment_t[t] + model.P_charge_discharge_with_eff_t[t])
        
    # Battery dynamics with efficiency charge =! discharge
    for t in range(1,len(time)):

        model.battery_dynamics_constraint.add(
            model.E_SOC_t[t] == model.E_SOC_t[t-1] -\
                model.P_charge_discharge_with_eff_t[t] * dt)

    # Battery delta for battery operation constraints
    model.P_battery_delta = pyo.Var(
        model.IDX5, 
        domain=pyo.Reals, 
        bounds=(-1e12, 1e12)
    )

    model.P_battery_delta_abs_pw = pyo.Var(
        model.IDX5, 
        domain=pyo.Reals, 
        bounds=(0, 1e12)
    )
    
    for t in model.IDX5:
        model.power_constraint.add(
               model.P_battery_delta[t] == model.P_charge_discharge_with_eff_t[t+1] - \
                                           model.P_charge_discharge_with_eff_t[t] )

    model.P_battery_delta_abs_pw_constr = pyo.Piecewise(
        model.IDX5,
        model.P_battery_delta_abs_pw, 
        model.P_battery_delta, 
        pw_pts=[-1e12,0,1e12], 
        f_rule=[1e12,0,1e12], 
        pw_constr_type='EQ', 
        #pw_repn='SOS2',
        #pw_repn='CC',
        #pw_repn='DCC',   
        pw_repn='DLOG',
        force_pw=False,
        warn_domain_coverage=False,
        #unbounded_domain_var=True
        )

    # Objective Function ##
    model.OBJ = pyo.Objective( expr = 
        # revenues and OPEX
        sum(price_ts[t] * model.P_HPP_t[t] for t in model.IDX1) - model.penalty[0]+\
        - sum(model.P_battery_delta_abs_pw[t]*cost_of_battery_P_fluct_in_peak_price_ratio*price_ts_to_max[t]
          for t in model.IDX5),
        sense=pyo.maximize 
        )

    opt = pyo.SolverFactory('glpk')
    #opt.options['tmlim'] = 60
    results = opt.solve(model, tee=True)
    results.write()
    #print('model.penalty[0]()',model.penalty[0]() )
    #print('\n\n')

    ## Return calculated results ##
    P_curtailment_ts = []
    P_HPP_ts = []    
    P_charge_discharge_ts = []
    E_SOC_ts = []

    for count in range(len(time)):
        P_curtailment_ts.append(model.P_curtailment_t[count]())
        P_HPP_ts.append(model.P_HPP_t[count]())
        P_charge_discharge_ts.append(model.P_charge_discharge_with_eff_t[count]())
        E_SOC_ts.append(model.E_SOC_t[count]())
    
    if model.penalty[0]() == None:
        penalty_ts = np.zeros(N_t) 
    else:
        penalty_ts = np.ones(N_t) * (model.penalty[0]()/N_t)


    return np.array(P_HPP_ts), np.array(P_curtailment_ts), np.array(P_charge_discharge_ts), np.array(E_SOC_ts), np.array(penalty_ts)

def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N:end]]
        counter += 1
    return batch

def ems_Wind_Solar_Battery_Pyomo(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,
    batch_size = 7*24,
):
    
    # split in batches, ussually a week
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2]+batches_all[-1]
    
    # allocate vars
    P_HPP_ts = np.array([]) 
    P_curtailment_ts = np.array([])
    P_charge_discharge_ts = np.array([]) 
    E_SOC_ts = np.array([])
    penalty_ts = np.array([])
    
    #print('\n\nEMS solved with pyomo\n')
    for ib, batch in enumerate(batches):
        wind_ts_sel = wind_ts.iloc[batch]
        solar_ts_sel = solar_ts.iloc[batch]
        price_ts_sel = price_ts.iloc[batch]
        E_batt_MWh_t_sel = E_batt_MWh_t.iloc[batch]
        
        #print(f'batch {ib+1} out of {len(batches)}')
        P_HPP_ts_batch, P_curtailment_ts_batch, P_charge_discharge_ts_batch,\
        E_SOC_ts_batch, penalty_batch = ems_Wind_Solar_Battery_Pyomo_parts(
            wind_ts = wind_ts_sel,
            solar_ts = solar_ts_sel,
            price_ts = price_ts_sel,
            P_batt_MW = P_batt_MW,
            E_batt_MWh_t = E_batt_MWh_t_sel,
            hpp_grid_connection = hpp_grid_connection,
            battery_depth_of_discharge = battery_depth_of_discharge,
            charge_efficiency = charge_efficiency,
            peak_hr_quantile = peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
        )
        
        P_HPP_ts = np.append(P_HPP_ts, P_HPP_ts_batch)
        P_curtailment_ts = np.append(
            P_curtailment_ts, P_curtailment_ts_batch)
        P_charge_discharge_ts = np.append(
            P_charge_discharge_ts,P_charge_discharge_ts_batch)
        E_SOC_ts = np.append(E_SOC_ts, E_SOC_ts_batch)
        penalty_ts = np.append(penalty_ts, penalty_batch)
        
    E_SOC_ts = np.append(E_SOC_ts, E_SOC_ts[0])
    
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
            
def operation_solar_batt_deg(
    pv_degradation,
    batt_degradation,
    wind_t,
    solar_t,
    hpp_curt_t,
    b_t,
    b_E_SOC_t,
    G_MW,
    b_E,
    battery_depth_of_discharge,
    battery_charge_efficiency,
    price_ts,
    b_E_SOC_0 = None,
    peak_hr_quantile = 0.9,
    n_full_power_hours_expected_per_day_at_peak_price = 3,
):

    """EMS operation for degraded PV and battery based on an existing EMS"""
    
    solar_deg_t_sat = solar_t * pv_degradation
    solar_deg_t_sat_loss = solar_t * (1 - pv_degradation)
    hpp_curt_t_deg = hpp_curt_t 
    P_loss =  np.maximum( 0 , solar_deg_t_sat_loss  - hpp_curt_t_deg)
    b_t_less_sol = b_t.copy()
    dt = 1
    # Reduction in power to battery due to reduction of solar
    for i in range(len(b_t)):
        if b_t[i] < 0:
            b_t_less_sol[i] = np.minimum(b_t[i] + P_loss[i],0)
    # Initialize the SoC
    b_E_SOC_t_sat = np.append(b_E , b_E_SOC_t.copy() )
    if b_E_SOC_0 == None:
        b_E_SOC_t_sat[0]= b_E_SOC_t[0]
    else:
        b_E_SOC_t_sat[0]= b_E_SOC_0
    # Update the SoC
    for i in range(len(b_t_less_sol)):
        if b_t_less_sol[i] < 0:
            b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt * battery_charge_efficiency
        if b_t_less_sol[i] >= 0 :
            b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt / battery_charge_efficiency
        b_E_SOC_t_sat[i+1] = np.clip(
            b_E_SOC_t_sat[i+1], 
            (1-battery_depth_of_discharge) * b_E * batt_degradation, b_E * batt_degradation  )
    # Recompute the battery power
    b_t_sat = b_t.copy()
    for i in range(len(b_t_sat)):
        if b_t[i] < 0:
            b_t_sat[i] = ( ( b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] ) / battery_charge_efficiency ) / dt
        elif b_t[i] >= 0:
            b_t_sat[i] = ( (b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] )  * battery_charge_efficiency ) / dt 
    H0_deg = wind_t + solar_t * pv_degradation
    Hpp_deg = np.minimum( wind_t + solar_t * pv_degradation + b_t_sat, G_MW)
    P_curt_deg = np.maximum( wind_t + solar_t * pv_degradation + b_t_sat - G_MW, 0)
    
    # Penalty
    N_t = len(price_ts) 
    N_days = N_t/24
    e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price*G_MW 
    e_peak_period_expected = e_peak_day_expected*N_days
    price_peak = np.quantile(price_ts, peak_hr_quantile)
    peak_hours_index = np.where(price_ts>=price_peak)[0]
    
    # print('len(Hpp_deg):', len(Hpp_deg))
    # print('peak_hours_index[0]:', peak_hours_index[0])
    # print('peak_hours_index[-1]:', peak_hours_index[-1])
    # print('N_t',N_t)
    
    e_penalty = e_peak_period_expected - np.sum([Hpp_deg[i] for i in peak_hours_index]) 
    penalty = price_peak*np.maximum(0, e_penalty)
    penalty_ts = np.ones_like(N_t) * (penalty/N_t)
    
    return Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat, penalty_ts


