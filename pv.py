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

# pvlib imports
from pvlib import pvsystem, tools, irradiance, atmosphere
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.tracking import SingleAxisTracker
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

class pvp(om.ExplicitComponent):
    """pv plant model"""

    def __init__(self,
                 weather_fn,
                 N_time,
                 latitude,
                 longitude,
                 altitude,
                 tracking = 'single_axis'):
        super().__init__()
        self.weather_fn = weather_fn
        self.N_time = N_time
        self.tracking = tracking

        pvloc = Location(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            name='Plant')

        weather = pd.read_csv(
            weather_fn, 
            index_col=0,
            parse_dates=True)

        weather['temp_air'] = weather['temp_air_1'] - 273.15  # Celcium
        weather['wind_speed'] = weather['WS_1']

        self.weather = weather
        self.pvloc = pvloc

    def setup(self):
        self.add_input(
            'surface_tilt',
            val=20,
            desc="Solar PV tilt angle in degs")

        self.add_input(
            'surface_azimuth',
            val=180,
            desc="Solar PV azimuth angle in degs, 180=south facing")

        self.add_input(
            'solar_MW',
            val=1,
            desc="Solar PV plant installed capacity",
            units='MW')
        
        self.add_output(
            'solar_t',
            desc="PV power time series",
            units='MW',
            shape=[self.N_time])
        
    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):
        
        # Sandia
        sandia_modules = pvsystem.retrieve_sam('SandiaMod')
        module_name = 'Canadian_Solar_CS5P_220M___2009_'
        module = sandia_modules[module_name]
        module['aoi_model'] = irradiance.aoi

        # 2. Inverter
        # -------------
        inverters = pvsystem.retrieve_sam('cecinverter')
        inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        
        temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        
        surface_tilt = inputs['surface_tilt']
        surface_azimuth = inputs['surface_azimuth']

        solar_MW = inputs['solar_MW'][0]
        
        if self.tracking == 'single_axis':
            system = SingleAxisTracker(
                module_parameters=module,
                inverter_parameters=inverter,
                temperature_model_parameters=temp_model,
                axis_tilt=surface_tilt[0],
                axis_azimuth=surface_azimuth[0],
                max_angle=90,
                backtrack=True,
                gcr=0.2857142857142857)
        else:
            system = PVSystem(
                module_parameters=module,
                inverter_parameters=inverter,
                temperature_model_parameters=temp_model,
                surface_tilt=surface_tilt,
                surface_azimuth=surface_azimuth)

        mc = ModelChain(system, self.pvloc)

        # Run solar with the WRF weather
        mc.run_model(self.weather)

        solar_t = (mc.ac / inverter.Paco)
        solar_t[solar_t>1] = 1
        solar_t[solar_t<0] = 0

        outputs['solar_t'] = solar_MW * solar_t.fillna(0.0)


class pvp_degradation_linear(om.ExplicitComponent):
    """docstring for pvp_degradation"""
    def __init__(
        self, 
        life_h = 25*365*24, 
        ):

        super().__init__()
        self.life_h = life_h
        
    def setup(self):
        self.add_input(
            'pv_deg_per_year',
            desc="PV degradation per year",
            val=0.5 / 100)
        
        self.add_output(
            'SoH_pv',
            desc="PV state of health time series",
            shape=[self.life_h])   

    def compute(self, inputs, outputs):

        pv_deg_per_year = inputs['pv_deg_per_year']
        
        t_over_year = np.arange(self.life_h)/(365*24)
        degradation = pv_deg_per_year * t_over_year

        y = 1 - degradation
        while len(y[y < 0]) > 0:
            #y[y<0] = 1 + y[y<0] - y[y<0][0]
            y[y < 0] = 0  # No replacement of PV panels

        outputs['SoH_pv'] = y     

class shadow(om.ExplicitComponent):
    """pv loss model due to shadows of wt"""

    # TODO implement degradation model in pcw
    # 1. Add input for:
    #    - turbine locations x_wt, y_wt in lat long
    #    - Pv locations
    #    - Altitude at the site
    # 2. Compute sun poisition:
    #    - sun position
    #    - simple wt shadow model to estimate covered area
    # 3. Estimate efficiency_t due to shadows

    def __init__(self, N_time):
        super().__init__()
        self.N_time = N_time

    def setup(self):
        self.add_input('solar_deg_t',
                       desc="PV power time series with degradation",
                       units='W',
                       shape=[self.N_time])
        self.add_output(
            'solar_deg_shad_t',
            desc="PV power time series with degradation and shadow losses",
            units='W',
            shape=[
                self.N_time])

    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):

        solar_deg_t = inputs['solar_deg_t']
        outputs['solar_deg_shad_t'] = solar_deg_t

