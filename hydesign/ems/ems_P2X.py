# %%

# import glob
# import os
# import time
# import copy

# basic libraries
import numpy as np
# from numpy import newaxis as na
import pandas as pd
import openmdao.api as om
# import yaml

# import xarray as xr
from docplex.mp.model import Model
from hydesign.ems.ems import expand_to_lifetime, split_in_batch

class ems_P2X(om.ExplicitComponent):
    """Energy management optimization model for HPP with P2X
    The energy management system optimization model consists in maximizing the revenue generated by the plant over a period of time,
    including a possible penalty for not meeting the requirement of energy generation during peak hours over the period. It also assigns
    a cost for rapid fluctuations of the battery in order to slow down its degradation.
    The EMS type is a CPLEX optimization.

    Parameters
    ----------
    wind_t : WPP power time series [MW]
    solar_t : PVP power time series [MW]
    price_t : Electricity price time series 
    b_P : Battery power capacity [MW]
    b_E : Battery energy storage capacity [MW]
    G_MW : Grid capacity [MW]
    battery_depth_of_discharge : battery depth of discharge
    battery_charge_efficiency : Wake affected power curve
    peak_hr_quantile : Quantile of price time series to define peak price hours (above this quantile)
    cost_of_battery_P_fluct_in_peak_price_ratio : cost of battery power fluctuations computed as a peak price ratio
    n_full_power_hours_expected_per_day_at_peak_price : Penalty occurs if number of full power hours expected per day at peak price are not reached
    price_H2: Price of Hydrogen
    ptg_MW: Electrolyzer power capacity
    storage_eff: Compressor efficiency for hydrogen storage
    ptg_deg: Electrolyzer rate of degradation annually
    hhv: High heat value
    m_H2_demand_t: Hydrogen demand times series
    HSS_kg: Hydrogen storage system capacity
    penalty_factor_H2: Penalty for not meeting hydrogen demand in an hour

    Returns
    -------
    wind_t_ext : WPP power time series
    solar_t_ext : PVP power time series
    price_t_ext : Electricity price time series
    hpp_t : HPP power time series
    hpp_curt_t : HPP curtailed power time series
    b_t : Battery charge/discharge power time series
    b_E_SOC_t : Battery energy SOC time series
    penalty_t : Penalty for not reaching expected energy productin at peak hours
    P_ptg_t: Electrolyzer power consumption time series
    m_H2_t: Hydrogen production time series
    m_H2_demand_t_ext: Hydrogen demand times series
    m_H2_demand_t: Hydrogen offtake times series
    LoS_H2_t: H2 storage level time series
    """

    def __init__(
        self, 
        N_time, 
        eff_curve,
        life_h = 25*365*24, 
        ems_type='cplex'):

        super().__init__()
        self.N_time = int(N_time)
        self.eff_curve = eff_curve
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
            desc="Penalty occurs if nunmber of full power hours expected per day at peak price are not reached.")
        self.add_input(
            'price_H2',
            desc="H2 price")
        self.add_input(
            'ptg_MW',
            desc="Electrolyzer power capacity.",
            units='MW')
        self.add_input(
            'storage_eff',
            desc="Compressor efficiency for hydrogen storage.")
        self.add_input(
            'ptg_deg',
            desc="Electrolyzer rate of degradation annually.")
        self.add_input(
            'hhv',
            desc="High heat value.")
        self.add_input(
            'm_H2_demand_t',
            desc="Hydrogen demand times series.",
            units='kg',
            shape=[self.N_time])
        self.add_input(
            'HSS_kg',
            desc="Hydrogen storgae capacity",
            units='kg')
        self.add_input(
            'penalty_factor_H2',
            desc="Penalty for not meeting hydrogen demand in an hour")
        
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
        self.add_output(
            'P_ptg_t',
            desc="Electrolyzer power consumption time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'm_H2_t',
            desc="Hydrogen production time series",
            units='kg',
            shape=[self.life_h])
        self.add_output(
            'm_H2_offtake_t',
            desc="Hydrogen offtake time series",
            units='kg',
            shape=[self.life_h])
        self.add_output(
            'LoS_H2_t',
            desc="H2 storage level time series",
            units='kg',
            shape=[self.life_h])
        self.add_output(
            'total_curtailment',
            desc="total curtailment in the lifetime",
            units='GW*h')
        self.add_output(
            'm_H2_demand_t_ext',
            desc="Hydrogen demand times series.",
            units='kg',
            shape=[self.life_h])
        
    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):
        
        wind_t = inputs['wind_t']
        solar_t = inputs['solar_t']
        price_t = inputs['price_t']
        m_H2_demand_t = inputs['m_H2_demand_t']

        b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']
        HSS_kg = inputs['HSS_kg']

        if self.ems_type == 'cplex':
            ems_WSB = ems_cplex_P2X
        else:
            raise Warning("This class should only be used for ems_cplex_P2X")

        
        # Avoid running an expensive optimization based ems if there is no battery
        # if ( b_P <= 1e-2 ) or (b_E == 0):
          #  ems_WSB = ems_rule_based
    
        battery_depth_of_discharge = inputs['battery_depth_of_discharge']
        battery_charge_efficiency = inputs['battery_charge_efficiency']
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        cost_of_battery_P_fluct_in_peak_price_ratio = inputs['cost_of_battery_P_fluct_in_peak_price_ratio'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs[
            'n_full_power_hours_expected_per_day_at_peak_price'][0]
        price_H2 = inputs['price_H2'][0]
        ptg_MW = inputs['ptg_MW'][0]
        storage_eff = inputs['storage_eff'][0]
        ptg_deg = inputs['ptg_deg'][0]
        hhv = inputs['hhv'][0]
        penalty_factor_H2 = inputs['penalty_factor_H2'][0]
        # Build a sintetic time to avoid problems with time sereis 
        # indexing in ems
        WSPr_df = pd.DataFrame(
            index=pd.date_range(
                start='01-01-1991 00:00',
                periods=len(wind_t),
                freq='1h'))

        WSPr_df['wind_t'] = wind_t
        WSPr_df['solar_t'] = solar_t
        WSPr_df['price_t'] = price_t
        WSPr_df['m_H2_demand_t'] = m_H2_demand_t  
        WSPr_df['E_batt_MWh_t'] = b_E[0]
        WSPr_df['H2_storage_t'] = HSS_kg[0]
        
        #print(WSPr_df.head())

        P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, P_ptg_ts, E_SOC_ts, m_H2_ts, m_H2_offtake_ts, LoS_H2_ts, penalty_ts = ems_WSB(
            wind_ts = WSPr_df.wind_t,
            solar_ts = WSPr_df.solar_t,
            price_ts = WSPr_df.price_t,
            P_batt_MW = b_P[0],
            E_batt_MWh_t = WSPr_df.E_batt_MWh_t,
            hpp_grid_connection = G_MW[0],
            battery_depth_of_discharge = battery_depth_of_discharge[0],
            charge_efficiency = battery_charge_efficiency[0],
            price_H2 = price_H2,
            ptg_MW = ptg_MW,
            HSS_kg = HSS_kg,
            storage_eff = storage_eff,
            ptg_deg = ptg_deg,
            hhv = hhv,
            m_H2_demand_ts = WSPr_df.m_H2_demand_t,
            H2_storage_t = WSPr_df.H2_storage_t,
            penalty_factor_H2 =penalty_factor_H2,
            eff_curve=self.eff_curve,
            peak_hr_quantile = peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price
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
        outputs['P_ptg_t'] = expand_to_lifetime(
            P_ptg_ts, life_h = self.life_h)
        outputs['m_H2_t'] = expand_to_lifetime(
            m_H2_ts, life_h = self.life_h)
        outputs['m_H2_offtake_t'] = expand_to_lifetime(
            m_H2_offtake_ts, life_h = self.life_h)
        outputs['LoS_H2_t'] = expand_to_lifetime(
            LoS_H2_ts, life_h = self.life_h)
        outputs['total_curtailment'] = outputs['hpp_curt_t'].sum()
        outputs['m_H2_demand_t_ext'] = expand_to_lifetime(
            m_H2_demand_t, life_h = self.life_h)
        

def ems_cplex_P2X(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    price_H2,
    ptg_MW,
    HSS_kg,
    storage_eff,
    ptg_deg,
    hhv,
    m_H2_demand_ts,
    H2_storage_t,
    penalty_factor_H2,
    eff_curve,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,  
    batch_size = 1*24,
):
    
    # split in batches, ussually a week
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2]+batches_all[-1]
    
    # allocate vars
    P_HPP_ts = np.zeros(len(wind_ts))
    P_curtailment_ts = np.zeros(len(wind_ts))
    P_charge_discharge_ts = np.zeros(len(wind_ts))
    P_ptg_ts = np.zeros(len(wind_ts))
    m_H2_ts = np.zeros(len(wind_ts))
    m_H2_offtake_ts = np.zeros(len(wind_ts))
    LoS_H2_ts = np.zeros(len(wind_ts))
    E_SOC_ts = np.zeros(len(wind_ts)+1)
    penalty_ts = np.zeros(len(wind_ts))
    
    for ib, batch in enumerate(batches):
        wind_ts_sel = wind_ts.iloc[batch]
        solar_ts_sel = solar_ts.iloc[batch]
        price_ts_sel = price_ts.iloc[batch]
        E_batt_MWh_t_sel = E_batt_MWh_t.iloc[batch] 
        m_H2_demand_ts_sel = m_H2_demand_ts.iloc[batch]
        H2_storage_t_sel = H2_storage_t.iloc[batch]
            
        #print(f'batch {ib+1} out of {len(batches)}')
        P_HPP_ts_batch, P_curtailment_ts_batch, P_charge_discharge_ts_batch, P_ptg_ts_batch,\
        E_SOC_ts_batch, m_H2_ts_batch, m_H2_offtake_ts_batch, LoS_H2_ts_batch, penalty_batch = ems_cplex_parts_P2X(
            wind_ts = wind_ts_sel,
            solar_ts = solar_ts_sel,
            price_ts = price_ts_sel,
            P_batt_MW = P_batt_MW,
            E_batt_MWh_t = E_batt_MWh_t_sel,
            hpp_grid_connection = hpp_grid_connection,
            battery_depth_of_discharge = battery_depth_of_discharge,
            charge_efficiency = charge_efficiency,
            price_H2=price_H2,
            ptg_MW=ptg_MW,
            HSS_kg=HSS_kg,
            storage_eff=storage_eff,
            ptg_deg=ptg_deg,
            hhv=hhv,
            m_H2_demand_ts = m_H2_demand_ts_sel,
            H2_storage_t = H2_storage_t_sel,
            penalty_factor_H2 = penalty_factor_H2,
            eff_curve = eff_curve,
            peak_hr_quantile = peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,      
        )
        
        # print()
        # print()
        # print()
        # print(ib, len(batch))
        # print()
        # print('len(wind_ts_sel)',len(wind_ts_sel))
        # print('len(P_HPP_ts_batch)',len(P_HPP_ts_batch))
        # print('len(P_curtailment_ts_batch)',len(P_curtailment_ts_batch))
        # print('len(P_charge_discharge_ts_batch)',len(P_charge_discharge_ts_batch))
        # print('len(E_SOC_ts_batch)',len(E_SOC_ts_batch))
        # print('len(penalty_batch)',len(penalty_batch))
        
        P_HPP_ts[batch] = P_HPP_ts_batch
        P_curtailment_ts[batch] = P_curtailment_ts_batch
        P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        E_SOC_ts[batch] = E_SOC_ts_batch[:-1]
        penalty_ts[batch] = penalty_batch
        P_ptg_ts[batch] = P_ptg_ts_batch
        m_H2_ts[batch] = m_H2_ts_batch
        m_H2_offtake_ts[batch] = m_H2_offtake_ts_batch
        LoS_H2_ts[batch] = LoS_H2_ts_batch
        
    E_SOC_ts[-1] = E_SOC_ts[0]
    LoS_H2_ts[-1] = LoS_H2_ts[0]
    
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, P_ptg_ts, E_SOC_ts, m_H2_ts, m_H2_offtake_ts, LoS_H2_ts, penalty_ts


def ems_cplex_parts_P2X(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    price_H2,
    ptg_MW,
    HSS_kg,
    storage_eff,
    ptg_deg,
    hhv,
    m_H2_demand_ts,
    H2_storage_t,
    penalty_factor_H2,
    eff_curve,
    peak_hr_quantile = 0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio = 0.5, #[0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price = 3,
):
    """EMS solver implemented in cplex

    Parameters
    ----------
    wind_ts : WPP power time series
    solar_ts : PVP power time series
    price_ts : price time series
    P_batt_MW : battery power
    E_batt_MWh_t : battery energy capacity time series
    H2_storage_t : hydrogen storgae capacity time series
    hpp_grid_connection : grid connection
    battery_depth_of_discharge : battery depth of discharge
    charge_efficiency : battery charge efficiency
    peak_hr_quantile : quantile of price time series to define peak price hours
    cost_of_battery_P_fluct_in_peak_price_ratio : cost of battery power fluctuations computed as a peak price ratio
    n_full_power_hours_expected_per_day_at_peak_price : Penalty occurs if number of full power hours expected per day at peak price are not reached
    price_H2: Price of Hydrogen
    ptg_MW: Electrolyzer power capacity
    HSS_kg: Hydrogen storage capacity
    storage_eff: Compressor efficiency for hydrogen storage
    ptg_deg: Electrolyzer rate of degradation annually
    hhv: High heat value
    m_H2_demand_ts: Hydrogen demand times series 
    penalty_factor_H2: Penalty on not meeting hydrogen demand in an hour

    Returns
    -------
    P_HPP_ts: HPP power time series
    P_curtailment_ts: HPP curtailed power time series
    P_charge_discharge_ts: Battery charge/discharge power time series 
    E_SOC_ts: Battery energy SOC time series 
    penalty_ts: penalty time series for not reaching expected energy production at peak hours
    P_ptg_ts: Electrolyzer power consumption time series
    m_H2_ts: Hydrogen production time series
    m_H2_offtake_ts: Hydrogen offtake time series
    LoS_H2_ts: Level of Hydrogen storage time series

    """
    
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
    mdl.context.cplex_parameters.emphasis.mip = 1 
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
    
    # Hydrogen storgae level
    LoS_H2_t = mdl.continuous_var_dict(
        SOCtime, lb=0, 
        name='Hydrogen storage level')
    
    # Power to gas plant power consumption, produced hydrogen, electrolyzer efficiency
    P_ptg_t = mdl.continuous_var_dict(
      time, lb=0, ub=ptg_MW,
      name = "Power to gas plant consumption"
      )
        
    m_H2_t = mdl.continuous_var_dict(time, lb = 0, name = 'Produced hydrogen')
    m_H2_offtake_t = mdl.continuous_var_dict(time, lb = 0, name = 'Hydrogen offtake')
        
    penalty = mdl.continuous_var(name='penalty', lb=-1e12)
    e_penalty = mdl.continuous_var(name='e_penalty', lb=-1e12)
    
    # Piecewise function for "absolute value" function
    fabs = mdl.piecewise(-1, [(0,0)], 1)
    
    mdl.maximize(
        # revenues and OPEX
        mdl.sum(
            price_ts[t] * P_HPP_t[t] + price_H2  *  m_H2_offtake_t[t] - penalty_factor_H2 *(m_H2_demand_ts[t]-m_H2_offtake_t[t])
            for t in time) - penalty \
        # Add cost for rapid charge-discharge for limiting the battery life use
        - mdl.sum(
           fabs(P_charge_discharge[t + pd.Timedelta('1hour')] - \
                P_charge_discharge[t])*cost_of_battery_P_fluct_in_peak_price_ratio*price_ts_to_max[t]
           for t in time[:-1])  
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
    
    mdl.add_constraint( LoS_H2_t[SOCtime[0]] == 0 )
    mdl.add_constraint( LoS_H2_t[SOCtime[-1]] == 0 )
#     # Intitial and end LoS_H2
#     mdl.add_constraint( LoS_H2_t[SOCtime[0]] == 0.5 * H2_storage_t[time[0]] )
    
#     # LoS_H2 at the end of the year has to be equal to LoS_H2 at the beginning of the year
#     mdl.add_constraint( LoS_H2_t[SOCtime[-1]] == 0.5 * H2_storage_t[time[0]] )

    # piecewise linear representation of battery charge vs dischrage effciency 
    f2 = mdl.piecewise(charge_efficiency,[(0,0)],1/charge_efficiency)
    
    # piecewise linear representation of H2 storage vs offtake effciency 
    f3 = mdl.piecewise(1/storage_eff,[(0,0)],storage_eff)
        
    # Caclulating electrolyzer efficiency as a function of load (piecewise linear approximation)
    eff_curve_list = [(load * ptg_MW, load * ptg_MW * efficiency) for load, efficiency in eff_curve]
    PEM = mdl.piecewise(0, eff_curve_list, 0)
    
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
            P_charge_discharge[t] - P_ptg_t[t])
        
        # charge/dischrage equation
        mdl.add_constraint(
            E_SOC_t[tt] == E_SOC_t[t] - 
            f2(P_charge_discharge[t]) * dt)
        
        # Hydrogen storgae equation
        mdl.add_constraint(
            LoS_H2_t[tt] == LoS_H2_t[t] + 
            f3(m_H2_t[t] - m_H2_offtake_t[t]))
        
        # Constraining battery energy level to minimum battery level
        mdl.add_constraint(
            E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t]
        )
        
        # Constraining battery energy level to maximum battery level
        mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])

        # Battery charge/discharge within its power rating
        mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW*charge_efficiency)
        mdl.add_constraint(P_charge_discharge[t] >= -P_batt_MW/charge_efficiency)
        
        # Constraining hydrogen offtake as per the demand time series and level of storage
        mdl.add_constraint(LoS_H2_t[t] <= H2_storage_t[t])
        mdl.add_constraint( m_H2_offtake_t[t] <= m_H2_demand_ts[t])
        
        #mdl.add_constraint( m_H2_offtake_t[t] <= LoS_H2_t[t])
        # when the H2 offtake is infinite, there is no storage, then H2_offtake is same as H2_produced
        if H2_storage_t[t] == 0:
            m_H2_offtake_t[t] = m_H2_t[t]
            
        # Calculating Hydrogen production with electrolyzer efficiency curve
        #mdl.add_constraint(m_H2_t[t] == AE(P_ptg_t[t])* storage_eff / hhv * 1000  * ptg_deg) 
        mdl.add_constraint(m_H2_t[t] == PEM(P_ptg_t[t])* storage_eff / hhv * 1000  * ptg_deg)        
        # Calculating Hydrogen production with constant electrolyzer efficiency
        # mdl.add_constraint(m_H2_t[t] == 0.65*P_ptg_t[t]* storage_eff / hhv * 1000  * ptg_deg)

    # Solving the problem
    sol = mdl.solve(
        log_output=False)
        #log_output=True)

    
    #print(mdl.export_to_string())
    #sol.display() 
    
    P_HPP_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_HPP_t), orient='index').loc[:,0]

    P_curtailment_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_curtailment_t), orient='index').loc[:,0]

    P_charge_discharge_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_charge_discharge), orient='index').loc[:,0]

    E_SOC_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(E_SOC_t), orient='index').loc[:,0]
    
    P_ptg_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_ptg_t), orient='index').loc[:,0]
    
    m_H2_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(m_H2_t), orient='index').loc[:,0]
    
    LoS_H2_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(LoS_H2_t), orient='index').loc[:,0]
    
    m_H2_offtake_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(m_H2_offtake_t), orient='index').loc[:,0]
  
    
    #make a time series like P_HPP with a constant penalty 
    penalty_2 = sol.get_value(penalty)
    penalty_ts = np.ones(N_t) * (penalty_2/N_t)
    
    mdl.end()
    
    # Cplex sometimes returns missing values :O
    P_HPP_ts = P_HPP_ts_df.reindex(time,fill_value=0).values
    P_curtailment_ts = P_curtailment_ts_df.reindex(time,fill_value=0).values
    P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time,fill_value=0).values
    E_SOC_ts = E_SOC_ts_df.reindex(SOCtime,fill_value=0).values
    P_ptg_ts = P_ptg_ts_df.reindex(time,fill_value=0).values
    m_H2_ts = m_H2_ts_df.reindex(time,fill_value=0).values
    LoS_H2_ts = LoS_H2_ts_df.reindex(time,fill_value=0).values
    m_H2_offtake_ts = m_H2_offtake_ts_df.reindex(time,fill_value=0).values

    if len(P_HPP_ts_df) < len(wind_ts):
        #print('recomputing p_hpp')
        P_HPP_ts = wind_ts.values + solar_ts.values +\
            - P_curtailment_ts + P_charge_discharge_ts - P_ptg_ts
    
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, P_ptg_ts, E_SOC_ts, m_H2_ts, m_H2_offtake_ts, LoS_H2_ts, penalty_ts


