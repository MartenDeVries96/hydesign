# %%
import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import pandas as pd
import xarray as xr
import openmdao.api as om

class genericWT_surrogate(om.ExplicitComponent):
    """
    Metamodel of the wind turbine.

    It relies on a look-up table (genWT_fn) of the WT performance for different 
    specific powers (sp=p_rated/rotor_area [W/m2]). 
    
    WT performance is: 
        (1) power vs hub height ws curve 
        (2) thurst coefficient vs hub heigh ws curve.

    Parameters
    ----------
    Turbine's hub height : the hub height of the wind turbine
    Turbine's diameter : the diameter of the blade
    Turbine's rated power : the rated power of the wind turbine

    Returns
    -------
    Turbine's ws : wind speed points in the power curve
    Turbine's power curve : power curve of the wind turbine 
    Turbine's ct curve : ct curve of the wind turbine
    
    """

    def __init__(
        self, 
        genWT_fn='./look_up_tables/genWT.nc',
        N_ws = 51,
        ):
        super().__init__()
        self.genWT_fn = genWT_fn
        # number of points in the power curves
        self.N_ws = N_ws

    def setup(self):
        self.add_input('hh',
                       desc="Turbine's hub height",
                       units='m')
        self.add_input('d',
                       desc="Turbine's diameter",
                       units='m')
        self.add_input('p_rated',
                       desc="Turbine's rated power",
                       units='MW')

        self.add_output('ws',
                        desc="Turbine's ws",
                        units='m/s',
                        shape=[self.N_ws])
        self.add_output('pc',
                        desc="Turbine's power curve",
                        units='MW',
                        shape=[self.N_ws])
        self.add_output('ct',
                        desc="Turbine's ct curve",
                        shape=[self.N_ws])

    def setup_partials(self):
        self.declare_partials(['pc', 'ct'], '*', method='fd')

    def compute(self, inputs, outputs):
        
        p_rated = inputs['p_rated']
        A = get_rotor_area(inputs['d'])
        sp = p_rated*1e6/A
        
        genWT = xr.open_dataset(self.genWT_fn).interp(
            sp=sp, 
            kwargs={"fill_value": 0}
            )
    
        ws = genWT.ws.values
        pc = genWT.pc.values
        ct = genWT.ct.values
        
        outputs['ws'] = ws 
        outputs['pc'] = pc
        outputs['ct'] = ct
        
        genWT.close()
    

class genericWake_surrogate(om.ExplicitComponent):
    """
    Generic wind farm wake model

    It relies on a look-up table of the wake losses for different wind farms
    parameters: 
        (1) WT specific power (sp=p_rated/rotor_area [W/m2])
        (2) Number of wind turbines
        (3) Wind farm installation density (wind_MW_per_km2) in [MW/km2]
    
    Parameters
    ----------
    Nwt : Number of wind turbines
    Awpp : Land use area of the wind power plant
    d : Turbine's diameter
    p_rated : Turbine's rated power
    ws : wind speed points in the power curve
    pc : Turbine's power curve
    ct : Turbine's Ct coefficient curve

    Returns
    -------
    pcw : Wake affected power curve

    """
    def __init__(
        self, 
        genWake_fn='./look_up_tables/genWake_upd.nc',
        N_ws = 51,
        ):

        super().__init__()
        self.genWake_fn = genWake_fn
        # number of points in the power curves
        self.N_ws = N_ws

    def setup(self):
        self.add_discrete_input(
            'Nwt',
            val=1,
            desc="Number of wind turbines")
        self.add_input(
            'Awpp',
            desc="Land use area of WPP",
            units='km**2')
        self.add_input(
            'd',
            desc="Turbine's diameter",
            units='m')
        self.add_input(
            'p_rated',
            desc="Turbine's rated power",
            units='MW')
        self.add_input(
            'ws',
            desc="Turbine's ws",
            units='m/s',
            shape=[self.N_ws])
        self.add_input(
            'pc',
            desc="Turbine's power curve",
            units='MW',
            shape=[self.N_ws])
        self.add_input(
            'ct',
            desc="Turbine's ct curve",
            shape=[self.N_ws])

        self.add_output(
            'pcw',
            desc="Wake affected power curve",
            shape=[self.N_ws])

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        ws = inputs['ws']
        Nwt = discrete_inputs['Nwt']
        Awpp = inputs['Awpp']  # in km2
        d = inputs['d']  # in m
        p_rated = inputs['p_rated']
        
        A = get_rotor_area(d)
        sp = p_rated*1e6/A
        wind_MW_per_km2 = Nwt*p_rated/(Awpp + 1e-10*(Awpp==0))
        
        genWake_sm = xr.open_dataset(self.genWake_fn).interp(
            sp=sp, 
            Nwt=Nwt, 
            wind_MW_per_km2=wind_MW_per_km2,
            kwargs={"fill_value": 1}
            )
        wl = genWake_sm.wl.values
        
        pcw = inputs['pc'] * (1 - wl)        
        outputs['pcw'] = pcw * Nwt * p_rated

        genWake_sm.close()

class wpp(om.ExplicitComponent):
    """
    Wind power plant model

    Provides the wind power time series using wake affected power curve and the wind speed time series.

    Parameters
    ----------
    ws : Turbine's ws
    pcw : Wake affected power curve
    wst : wind speed time series at the hub height

    Returns
    -------
    wind_t : power time series at the hub height

    """

    def __init__(
        self, 
        N_time,
        N_ws = 51,
        wpp_efficiency = 0.95,
        ):
        super().__init__()
        self.N_time = N_time
        # number of points in the power curves
        self.N_ws = N_ws
        self.wpp_efficiency = wpp_efficiency

    def setup(self):
        self.add_input('ws',
                       desc="Turbine's ws",
                       units='m/s',
                       shape=[self.N_ws])
        self.add_input('pcw',
                       desc="Wake affected power curve",
                       shape=[self.N_ws])
        self.add_input('wst',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.N_time])

        self.add_output('wind_t',
                        desc="power time series at the hub height",
                        units='MW',
                        shape=[self.N_time])


    def compute(self, inputs, outputs):

        ws = inputs['ws']
        pcw = inputs['pcw']
        wst = inputs['wst']

        outputs['wind_t'] = self.wpp_efficiency * np.interp(
            wst, ws, pcw, left=0, right=0, period=None)

# -----------------------------------------------------------------------
# Auxiliar functions 
# -----------------------------------------------------------------------        

def get_rotor_area(d): return np.pi*(d/2)**2
def get_rotor_d(area): return 2*(area/np.pi)**0.5
