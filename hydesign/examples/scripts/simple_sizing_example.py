import os
import time
import yaml
import pandas as pd

from hydesign.assembly.hpp_assembly_OR_tools import hpp_model
from hydesign.examples import examples_filepath

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
        input_ts_fn = input_ts_fn,
)

start = time.time()

clearance = 10
sp = 350
p_rated = 5
Nwt = 62
wind_MW_per_km2 = 7
solar_MW = 50
surface_tilt = 50
surface_azimuth = 180
solar_DCAC = 1.5
b_P = 20
b_E_h  = 3
cost_of_batt_degr = 5

x = [clearance, sp, p_rated, Nwt, wind_MW_per_km2, \
solar_MW, surface_tilt, surface_azimuth, solar_DCAC, \
b_P, b_E_h , cost_of_batt_degr]

outs = hpp.evaluate(*x)

hpp.print_design(x, outs)

end = time.time()
print(f'exec. time [min]:', (end - start)/60 )


def main():
    if __name__ == '__main__':
        from hydesign.assembly.hpp_assembly import hpp_model
        from hydesign.Parallel_EGO import get_kwargs, EfficientGlobalOptimizationDriver

        # Simple example to size wind only with a single core to run test machines and colab
        
        inputs = {
            'example': 4,
            'name': None,
            'longitude': None,
            'latitude': None,
            'altitude': None,
            'input_ts_fn': None,
            'sim_pars_fn': None,
    
            'opt_var': "NPV_over_CAPEX",
            'num_batteries': 1,
            'n_procs': 1,
            'n_doe': 8,
            'n_clusters': 1,
            'n_seed': 0,
            'max_iter': 2,
            'final_design_fn': 'hydesign_design_0.csv',
            'npred': 3e4,
            'tol': 1e-6,
            'min_conv_iter': 2,
            'work_dir': './',
            'hpp_model': hpp_model,
            }
    
        kwargs = get_kwargs(inputs)
        kwargs['variables'] = {
            'clearance [m]':
                {'var_type':'design',
                  'limits':[10, 60],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #   'value': 35
                #   },
             'sp [W/m2]':
                {'var_type':'design',
                 'limits':[200, 359],
                 'types':'int'
                 },
            'p_rated [MW]':
                {'var_type':'design',
                  'limits':[1, 10],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #  'value': 6
                 # },
            'Nwt':
                {'var_type':'design',
                  'limits':[0, 400],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #   'value': 200
                #   },
            'wind_MW_per_km2 [MW/km2]':
                {'var_type':'design',
                  'limits':[5, 9],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #   'value': 7
                #   },
            'solar_MW [MW]':
                # {'var_type':'design',
                #   'limits':[0, 400],
                #   'types':'int'
                #   },
                {'var_type':'fixed',
                  'value': 200
                  },
            'surface_tilt [deg]':
                # {'var_type':'design',
                #   'limits':[0, 50],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 25
                  },
            'surface_azimuth [deg]':
                # {'var_type':'design',
                #   'limits':[150, 210],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 180
                  },
            'DC_AC_ratio':
                # {'var_type':'design',
                #   'limits':[1, 2.0],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value':1.0,
                  },
            'b_P [MW]':
                # {'var_type':'design',
                #   'limits':[0, 100],
                #   'types':'int'
                #   },
                {'var_type':'fixed',
                  'value': 50
                  },
            'b_E_h [h]':
                # {'var_type':'design',
                #   'limits':[1, 10],
                #   'types':'int'
                #   },
                {'var_type':'fixed',
                  'value': 6
                  },
            'cost_of_battery_P_fluct_in_peak_price_ratio':
                # {'var_type':'design',
                #   'limits':[0, 20],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 10},
            }
        EGOD = EfficientGlobalOptimizationDriver(**kwargs)
        EGOD.run()
        result = EGOD.result

#main()