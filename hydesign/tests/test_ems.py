# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import numpy as np
import pandas as pd
import pytest
import pickle

from hydesign.tests.test_files import tfp
from hydesign.ems import ems_cplex, operation_solar_batt_deg

# ------------------------------------------------------------------------------------------------
def run_ems():
    with open(tfp+'ems_input_ems.pickle', 'rb') as f:
        input_ems = pickle.load(f)
    ems_out = ems_cplex(**input_ems)
    return ems_out

def load_ems():
    with open(tfp+'ems_output_ems.pickle','rb') as f:
        ems_out = pickle.load(f)
    return ems_out

def test_ems():
    ems_out = run_ems()
    ems_out_data = load_ems()
    for i in range(len(ems_out)):
        np.testing.assert_allclose(ems_out[i], ems_out_data[i])
        #print(np.allclose(ems_out[i], ems_out_data[i]))

# ------------------------------------------------------------------------------------------------
def run_operation_with_deg():
    with open(tfp+'ems_input_ems.pickle', 'rb') as f:
        input_ems = pickle.load(f)
    with open(tfp+'ems_output_ems.pickle','rb') as f:
        ems_out = pickle.load(f)
    P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts =  ems_out
    out_operation_with_deg = operation_solar_batt_deg(
                    pv_degradation = 0.8,
                    batt_degradation = 0.8,
                    wind_t = input_ems['wind_ts'].values,
                    solar_t = input_ems['solar_ts'].values,
                    hpp_curt_t = P_curtailment_ts,
                    b_t = P_charge_discharge_ts,
                    b_E_SOC_t = E_SOC_ts,
                    G_MW = input_ems['hpp_grid_connection'],
                    b_E = input_ems['E_batt_MWh_t'][0],
                    battery_depth_of_discharge = input_ems['battery_depth_of_discharge'],
                    battery_charge_efficiency = input_ems['charge_efficiency'],
                    b_E_SOC_0 = 0,
                    price_ts = input_ems['price_ts'],
                    peak_hr_quantile = input_ems['peak_hr_quantile'],
                    n_full_power_hours_expected_per_day_at_peak_price = \
                        input_ems['n_full_power_hours_expected_per_day_at_peak_price'],
                )
    return out_operation_with_deg

def load_operation_with_deg():
    with open(tfp+'out_operation_with_deg.pickle','rb') as f:
        out_operation_with_deg = pickle.load(f)
    return out_operation_with_deg

def test_operation_with_deg():
    out_operation_with_deg = run_operation_with_deg()
    out_operation_with_deg_data = load_operation_with_deg()
    for i in range(len(out_operation_with_deg)):
        np.testing.assert_allclose(out_operation_with_deg[i], out_operation_with_deg_data[i])
        #print(np.allclose(out_operation_with_deg[i], out_operation_with_deg_data[i]))

