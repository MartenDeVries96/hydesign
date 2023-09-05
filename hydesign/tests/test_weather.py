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
from hydesign.examples import examples_filepath
from hydesign.weather import interpolate_WS_loglog

# ------------------------------------------------------------------------------------------------
def run_interp_ws():
    hh = 100
    weather = pd.read_csv(examples_filepath+'Europe/GWA2/input_ts_Denmark_good_solar.csv',index_col=0)
    interp_ws_out = interpolate_WS_loglog(weather,hh)
    return interp_ws_out.WS.values, interp_ws_out.dWS_dz.values

def load_interp_ws():
    with open(tfp+'weather_output_interp_ws.pickle','rb') as f:
        interp_ws_out = pickle.load(f)
    return interp_ws_out

def test_interp_ws():
    interp_ws_out = run_interp_ws()
    interp_ws_out_data = load_interp_ws()
    for i in range(len(interp_ws_out)):
        np.testing.assert_allclose(interp_ws_out[i], interp_ws_out_data[i])
        # print(np.allclose(interp_ws_out[i], interp_ws_out_data[i]))