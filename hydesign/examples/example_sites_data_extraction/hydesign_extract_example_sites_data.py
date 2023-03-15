import hydesign
import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.hpp_assembly_simplified import hpp_model_simple
from hydesign.examples import examples_filepath

import argparse

if __name__ == "__main__":
    
    # -----------------------------------------------
    # Arguments from the outer .sh (shell) script
    # -----------------------------------------------
    parser=argparse.ArgumentParser()
    parser.add_argument('--example', default=None, 
                        help='ID (index) to run an example site, based on ./examples/examples_sites.csv')    
    args=parser.parse_args()
    
    example = args.example
    
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
    if example != None:
        examples_sites = examples_sites.iloc[[int(example)],:]

    
    for iloc in range(len(examples_sites.case)):
        ex_site = examples_sites.iloc[[iloc],:]

        name = ex_site['name'].values[0]
        case = ex_site['case'].values[0]
        longitude = ex_site['longitude'].values[0]
        latitude = ex_site['latitude'].values[0]
        altitude = ex_site['altitude'].values[0]

        sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
        price_fn = examples_filepath+ex_site['price_fn'].values[0]
        price = pd.read_csv(price_fn,index_col=0, parse_dates=True)[ex_site.price_col.values[0]]

        print()
        print()
        print(iloc, case, name)
        print()

        hpp = hpp_model_simple(
                latitude,
                longitude,
                altitude,
                # Dummy data
                rotor_diameter_m = 100,
                hub_height_m = 100,
                wt_rated_power_MW = 1,
                surface_tilt_deg = 30,
                surface_azimuth_deg = 180,
                DC_AC_ratio = 1.5,
                num_batteries = 1,
                work_dir = './',
                sim_pars_fn = sim_pars_fn,
                input_ts_fn=None,
                price_fn=price,
        )

        inputs = pd.read_csv('input_ts.csv', index_col=0, parse_dates=True)

        input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
        inputs.to_csv(input_ts_fn)

        os.remove('input_ts.csv') 