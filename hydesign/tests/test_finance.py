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
from hydesign.finance import calculate_NPV_IRR, calculate_WACC
from hydesign.examples import examples_filepath
from hydesign.hpp_assembly import hpp_model

# ------------------------------------------------------------------------------------------------
def run_WACC():
    with open(tfp+'finance_input_WACC.pickle', 'rb') as f:
        input_WACC = pickle.load(f)
    WACC_out = calculate_WACC(**input_WACC)
    return WACC_out

def load_WACC():
    with open(tfp+'finance_output_WACC.pickle','rb') as f:
        WACC_out = pickle.load(f)
    return WACC_out

def test_WACC():
    WACC_out = run_WACC()
    WACC_out_data = load_WACC()
    for i in range(len(WACC_out)):
        np.testing.assert_allclose(WACC_out[i], WACC_out_data[i])
        # print(np.allclose(WACC_out[i], WACC_out_data[i]))

# ------------------------------------------------------------------------------------------------
def run_NPV():
    with open(tfp+'finance_input_NPV.pickle', 'rb') as f:
        input_NPV = pickle.load(f)
    NPV_out = calculate_NPV_IRR(**input_NPV)
    return NPV_out

def load_NPV():
    with open(tfp+'finance_output_NPV.pickle','rb') as f:
        NPV_out = pickle.load(f)
    return NPV_out

def test_NPV():
    NPV_out = run_NPV()
    NPV_out_data = load_NPV()
    for i in range(len(NPV_out)):
        np.testing.assert_allclose(NPV_out[i], NPV_out_data[i])
        # print(np.allclose(NPV_out[i], NPV_out_data[i]))

# ------------------------------------------------------------------------------------------------
# Test CAPEX phasing
def get_CAPEX_phasing_model():
    name = 'Denmark_good_wind'
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]

    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]

    hpp = hpp_model(
            latitude,
            longitude,
            altitude,
            num_batteries = 3,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
    )

    x=[10, 350, 5, 60, 7, 0, 50, 180, 1.5, 0, 3, 5]
    return hpp, x, longitude, latitude, altitude

def run_CAPEX_phasing_model(capex_phasing = [[-1, 0, 5, 10, 15], [0, 0.25, 0.25, 0.25, 0.25]]):
    hpp, x, longitude, latitude, altitude = get_CAPEX_phasing_model()
    hpp.prob.model.finance.capex_phasing = capex_phasing
    outs = hpp.evaluate(*x)
    return hpp, x, outs, longitude, latitude, altitude
    
def save_CAPEX_phasing_model():
    hpp, x, outs, longitude, latitude, altitude = run_CAPEX_phasing_model()
    hpp.evaluation_in_csv(tfp+'capex_phasing', longitude, latitude, altitude, x, outs)

def test_CAPEX_phasing():
    hpp, x, outs, longitude, latitude, altitude = run_CAPEX_phasing_model()
    df = hpp.evaluation_in_csv(tfp+'tmp/capex_phasing', longitude, latitude, altitude, x, outs)
    df_ref = pd.read_csv(tfp+'capex_phasing.csv', index_col=0)
    np.testing.assert_allclose(df.to_numpy(dtype=float), df_ref.to_numpy(dtype=float))

def test_CAPEX_phasing_not_summing_to_1():
    with pytest.raises(AssertionError) as e_info:
        capex_phasing = [[-1, 0, 5, 10, 15], [0.25, 0.25, 0.25, 0.25, 0.25]]
        hpp, x, outs, longitude, latitude, altitude = run_CAPEX_phasing_model(capex_phasing=capex_phasing)
