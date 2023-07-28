import numpy as np
import pandas as pd
import pytest
import pickle

from hydesign.tests.test_files import tfp
from hydesign.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath

# ------------------------------------------------------------------------------------------------
# design 1

def run_evaluation_design_1():
    output_df = pd.read_csv(
        tfp+'France_good_wind_design.csv',
        index_col=0, 
        parse_dates = True)
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
    name = 'France_good_wind'
    ex_site = examples_sites.loc[examples_sites.name == name]
    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    hpp = hpp_model(
        latitude,
        longitude,
        altitude,
        num_batteries = 1,
        work_dir = './',
        sim_pars_fn = sim_pars_fn,
        input_ts_fn = input_ts_fn)
    clearance = output_df.loc['clearance [m]','Design 1']
    sp = output_df.loc['sp [W/m2]','Design 1']
    p_rated = output_df.loc['p_rated [MW]','Design 1']
    Nwt = output_df.loc['Nwt','Design 1']
    wind_MW_per_km2 = output_df.loc['wind_MW_per_km2 [MW/km2]','Design 1']
    solar_MW = output_df.loc['solar_MW [MW]','Design 1']
    surface_tilt = output_df.loc['surface_tilt [deg]','Design 1']
    surface_azimuth = output_df.loc['surface_azimuth [deg]','Design 1']
    solar_DCAC = output_df.loc['DC_AC_ratio','Design 1']
    b_P = output_df.loc['b_P [MW]','Design 1']
    b_E_h  = output_df.loc['b_E_h [h]','Design 1']
    cost_of_batt_degr = output_df.loc['cost_of_battery_P_fluct_in_peak_price_ratio','Design 1']

    x = [clearance, sp, p_rated, Nwt, wind_MW_per_km2, \
    solar_MW, surface_tilt, surface_azimuth, solar_DCAC, \
    b_P, b_E_h , cost_of_batt_degr]

    outs = hpp.evaluate(*x)
    
    return outs

def load_evaluation_design_1():
    output_df = pd.read_csv(
        tfp+'France_good_wind_design.csv',
        index_col=0, 
        parse_dates = True)
    load_file = np.array(output_df.iloc[15:]['Design 1'])
    
    return load_file

def test_evaluation_design_1():
    evaluation_metrics = run_evaluation_design_1()
    loaded_metrics = load_evaluation_design_1()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i])
        
# ------------------------------------------------------------------------------------------------
# design 2

def run_evaluation_design_2():
    output_df = pd.read_csv(
        tfp+'France_good_wind_design.csv',
        index_col=0, 
        parse_dates = True)
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
    name = 'France_good_wind'
    ex_site = examples_sites.loc[examples_sites.name == name]
    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    hpp = hpp_model(
        latitude,
        longitude,
        altitude,
        num_batteries = 1,
        work_dir = './',
        sim_pars_fn = sim_pars_fn,
        input_ts_fn = input_ts_fn)
    clearance = output_df.loc['clearance [m]','Design 2']
    sp = output_df.loc['sp [W/m2]','Design 2']
    p_rated = output_df.loc['p_rated [MW]','Design 2']
    Nwt = output_df.loc['Nwt','Design 2']
    wind_MW_per_km2 = output_df.loc['wind_MW_per_km2 [MW/km2]','Design 2']
    solar_MW = output_df.loc['solar_MW [MW]','Design 2']
    surface_tilt = output_df.loc['surface_tilt [deg]','Design 2']
    surface_azimuth = output_df.loc['surface_azimuth [deg]','Design 2']
    solar_DCAC = output_df.loc['DC_AC_ratio','Design 2']
    b_P = output_df.loc['b_P [MW]','Design 2']
    b_E_h  = output_df.loc['b_E_h [h]','Design 2']
    cost_of_batt_degr = output_df.loc['cost_of_battery_P_fluct_in_peak_price_ratio','Design 2']

    x = [clearance, sp, p_rated, Nwt, wind_MW_per_km2, \
    solar_MW, surface_tilt, surface_azimuth, solar_DCAC, \
    b_P, b_E_h , cost_of_batt_degr]

    outs = hpp.evaluate(*x)
    
    return outs

def load_evaluation_design_2():
    output_df = pd.read_csv(
        tfp+'France_good_wind_design.csv',
        index_col=0, 
        parse_dates = True)
    load_file = np.array(output_df.iloc[15:]['Design 2'])
    
    return load_file

def test_evaluation_design_2():
    evaluation_metrics = run_evaluation_design_2()
    loaded_metrics = load_evaluation_design_2()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i])
        
# ------------------------------------------------------------------------------------------------

# # # design 3

def run_evaluation_design_3():
    output_df = pd.read_csv(
        tfp+'France_good_wind_design.csv',
        index_col=0, 
        parse_dates = True)
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
    name = 'France_good_wind'
    ex_site = examples_sites.loc[examples_sites.name == name]
    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    hpp = hpp_model(
        latitude,
        longitude,
        altitude,
        num_batteries = 1,
        work_dir = './',
        sim_pars_fn = sim_pars_fn,
        input_ts_fn = input_ts_fn)
    clearance = output_df.loc['clearance [m]','Design 3']
    sp = output_df.loc['sp [W/m2]','Design 3']
    p_rated = output_df.loc['p_rated [MW]','Design 3']
    Nwt = output_df.loc['Nwt','Design 3']
    wind_MW_per_km2 = output_df.loc['wind_MW_per_km2 [MW/km2]','Design 3']
    solar_MW = output_df.loc['solar_MW [MW]','Design 3']
    surface_tilt = output_df.loc['surface_tilt [deg]','Design 3']
    surface_azimuth = output_df.loc['surface_azimuth [deg]','Design 3']
    solar_DCAC = output_df.loc['DC_AC_ratio','Design 3']
    b_P = output_df.loc['b_P [MW]','Design 3']
    b_E_h  = output_df.loc['b_E_h [h]','Design 3']
    cost_of_batt_degr = output_df.loc['cost_of_battery_P_fluct_in_peak_price_ratio','Design 3']

    x = [clearance, sp, p_rated, Nwt, wind_MW_per_km2, \
    solar_MW, surface_tilt, surface_azimuth, solar_DCAC, \
    b_P, b_E_h , cost_of_batt_degr]

    outs = hpp.evaluate(*x)
    
    return outs

def load_evaluation_design_3():
    output_df = pd.read_csv(
        tfp+'France_good_wind_design.csv',
        index_col=0, 
        parse_dates = True)
    load_file = np.array(output_df.iloc[15:]['Design 3'])
    
    return load_file



def test_evaluation_design_3():
    evaluation_metrics = run_evaluation_design_3()
    loaded_metrics = load_evaluation_design_3()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i])
        
# ------------------------------------------------------------------------------------------------
