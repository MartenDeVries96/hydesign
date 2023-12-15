# %%
import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import scipy as sp
import pandas as pd
import xarray as xr
import openmdao.api as om

from hydesign.look_up_tables import lut_filepath
from hydesign.ems import expand_to_lifetime

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
        genWT_fn = lut_filepath+'genWT_v3.nc',
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
        
        ws, pc, ct = get_WT_curves(
            genWT_fn=self.genWT_fn,
            specific_power=sp) 
        
        outputs['ws'] = ws 
        outputs['pc'] = pc
        outputs['ct'] = ct
    

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
        genWake_fn = lut_filepath+'genWake_v3.nc',
        N_ws = 51,
        ):

        super().__init__()
        self.genWake_fn = genWake_fn
        # number of points in the power curves
        self.N_ws = N_ws

    def setup(self):
        #self.add_discrete_input(
        self.add_input(
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

    def compute(self, inputs, outputs):#, discrete_inputs, discrete_outputs):

        ws = inputs['ws']
        pc = inputs['pc']
        Nwt = inputs['Nwt']
        #Nwt = discrete_inputs['Nwt']
        Awpp = inputs['Awpp']  # in km2
        d = inputs['d']  # in m
        p_rated = inputs['p_rated']
        
        A = get_rotor_area(d)
        sp = p_rated*1e6/A
        wind_MW_per_km2 = Nwt*p_rated/(Awpp + 1e-10*(Awpp==0))
        
        outputs['pcw'] = get_wake_affected_pc(
            genWake_fn = self.genWake_fn, 
            specific_power = sp,
            Nwt = Nwt,
            wind_MW_per_km2 = wind_MW_per_km2,
            ws = ws,
            pc = pc,
            p_rated = p_rated
        )

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

        outputs['wind_t'] = get_wind_ts(
            ws = ws,
            pcw = pcw,
            wst = wst,
            wpp_efficiency = self.wpp_efficiency,
        )

class wpp_with_degradation(om.ExplicitComponent):
    """
    Wind power plant model

    Provides the wind power time series using wake affected power curve and the wind speed time series.

    Parameters
    ----------
    N_time : Number of time-steps in weather simulation
    life_h : lifetime in hours
    N_ws : number of points in the power curves
    wpp_efficiency : WPP efficiency
    wind_deg_yr : year list for providing WT degradation curve
    wind_deg : degradation losses at yr
    share_WT_deg_types : share ratio between two degradation mechanism (0: only shift in power curve, 1: degradation as a loss factor )
    ws : Power curve wind speed list
    pcw : Wake affected power curve
    wst : wind speed time series at the hub height

    Returns
    -------
    wind_t_ext_deg : power time series with degradation extended through lifetime

    """

    def __init__(
        self, 
        N_time,
        N_ws = 51,
        wpp_efficiency = 0.95,
        life_h = 25*365*24,
        wind_deg_yr = [0, 25],
        wind_deg = [0, 25*1/100],
        share_WT_deg_types = 0.5,
        weeks_per_season_per_year = None,
        ):
        super().__init__()
        self.N_time = N_time
        self.life_h = life_h
        # number of points in the power curves
        self.N_ws = N_ws
        self.wpp_efficiency = wpp_efficiency
        
        # number of elements in WT degradation curve
        self.wind_deg_yr = wind_deg_yr
        self.wind_deg = wind_deg
        self.share_WT_deg_types = share_WT_deg_types

        # In case data is provided as weeks per season
        self.weeks_per_season_per_year = weeks_per_season_per_year
        
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

        self.add_output('wind_t_ext_deg',
                        desc="power time series with degradation",
                        units='MW',
                        shape=[self.life_h])


    def compute(self, inputs, outputs):
        
        ws = inputs['ws']
        pcw = inputs['pcw']
        wst = inputs['wst']

        wst_ext = expand_to_lifetime(
            wst, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        
        outputs['wind_t_ext_deg'] = self.wpp_efficiency*get_wind_ts_degradation(
            ws = ws, 
            pc = pcw, 
            ws_ts = wst_ext, 
            yr = self.wind_deg_yr, 
            wind_deg=self.wind_deg, 
            life_h = self.life_h, 
            share = self.share_WT_deg_types)

class existing_wpp(om.ExplicitComponent):
    """
    Wind power plant model for an existing layout

    Provides the wind power time series using wake affected power curve and the wind speed time series.

    Parameters
    ----------
    N_time : Number of time-steps in weather simulation
    existing_wpp_power_curve_xr_fn: File name of a netcdf xarray. 
            
            The xarray should include 'P_no_wake' as function of 'ws' and 'wake_losses' as a function of 'ws' and 'wd'.
            Note that the wd must include both 0 and 360, and a large WS (for interpolation). 
            Resolution of ws and wd is flexible.
            
            '''
            <xarray.Dataset>
            Dimensions:          (ws: 53, wd: 361)
            Coordinates:
              * ws               (ws) float64 0.0 0.5 1.0 1.5 2.0 ... 24.5 25.0 25.0 100.0
              * wd               (wd) float64 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0 360.0
            Data variables:
                wake_losses_eff  (ws, wd) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
                P_no_wake        (ws) float64 0.0 0.0 0.0 0.0 0.0 ... 100.0 100.0 0.0 0.0
            '''
            
    wpp_efficiency : WPP efficiency
    wst : wind speed time series at the hub height
    wdt : wind direction time series at the hub height

    Returns
    -------
    wind_t_ext_deg : power time series with degradation extended through lifetime

    """

    def __init__(
        self, 
        N_time,
        existing_wpp_power_curve_xr_fn, 
        wpp_efficiency = 0.95,
        ):
        
        super().__init__()
        self.N_time = N_time
        self.wpp_efficiency = wpp_efficiency

        self.existing_wpp_power_curve_xr_fn = existing_wpp_power_curve_xr_fn
        
        
    def setup(self):
        self.add_input('wst',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.N_time])
        self.add_input('wdt',
                       desc="wd time series at the hub height",
                       units='deg',
                       shape=[self.N_time])

        self.add_output('wind_t',
                        desc="power time series at the hub height",
                        units='MW',
                        shape=[self.N_time])


    def compute(self, inputs, outputs):

        N_time = self.N_time
        wpp_efficiency = self.wpp_efficiency
        
        wst = inputs['wst']
        wdt = inputs['wst']

        existing_wpp_power_curve_xr = xr.open_dataset(self.existing_wpp_power_curve_xr_fn)
        
        xr_time = xr.Dataset()
        xr_time['wst'] = xr.DataArray( 
            data=wst, 
            dims = ['t'],
            coords = {'t':np.arange(N_time)})
        xr_time['wdt'] = xr.DataArray( 
            data=wdt, 
            dims = ['t'],
            coords = {'t':np.arange(N_time)})

        wake_losses_eff_t = existing_wpp_power_curve_xr.wake_losses_eff.interp(ws=xr_time.wst, wd=xr_time.wdt).values

        wind_t_no_wake = get_wind_ts(
            ws = existing_wpp_power_curve_xr.ws.values,
            pcw = existing_wpp_power_curve_xr.P_no_wake.values,
            wst = wst,
            wpp_efficiency = self.wpp_efficiency,
        )
        
        outputs['wind_t'] = wake_losses_eff_t*wind_t_no_wake

class existing_wpp_with_degradation(om.ExplicitComponent):
    """
    Wind power plant model for an existing layout

    Provides the wind power time series using wake affected power curve and the wind speed time series.

    Parameters
    ----------
    N_time : Number of time-steps in weather simulation
    life_h : lifetime in hours
    existing_wpp_power_curve_xr_fn: File name of a netcdf xarray. 
            
            The xarray should include 'P_no_wake' as function of 'ws' and 'wake_losses' as a function of 'ws' and 'wd'.
            Note that the wd must include both 0 and 360, and a large WS (for interpolation). 
            Resolution of ws and wd is flexible.
            
            '''
            <xarray.Dataset>
            Dimensions:          (ws: 53, wd: 361)
            Coordinates:
              * ws               (ws) float64 0.0 0.5 1.0 1.5 2.0 ... 24.5 25.0 25.0 100.0
              * wd               (wd) float64 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0 360.0
            Data variables:
                wake_losses_eff  (ws, wd) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
                P_no_wake        (ws) float64 0.0 0.0 0.0 0.0 0.0 ... 100.0 100.0 0.0 0.0           
            '''
            
    wpp_efficiency : WPP efficiency
    wind_deg_yr : year list for providing WT degradation curve
    wind_deg : degradation losses at yr
    share_WT_deg_types : share ratio between two degradation mechanism (0: only shift in power curve, 1: degradation as a loss factor )
    ws : Power curve wind speed list
    pcw : Wake affected power curve
    wst : wind speed time series at the hub height

    Returns
    -------
    wind_t_ext_deg : power time series with degradation extended through lifetime

    """

    def __init__(
        self, 
        N_time,
        existing_wpp_power_curve_xr_fn, 
        wpp_efficiency = 0.95,
        life_h = 25*365*24,
        wind_deg_yr = [0, 25],
        wind_deg = [0, 25*1/100],
        share_WT_deg_types = 0.5,
        weeks_per_season_per_year = None,
        ):
        
        super().__init__()
        self.N_time = N_time
        self.life_h = life_h
        self.wpp_efficiency = wpp_efficiency

        self.existing_wpp_power_curve_xr_fn = existing_wpp_power_curve_xr_fn
        
        # number of elements in WT degradation curve
        self.wind_deg_yr = wind_deg_yr
        self.wind_deg = wind_deg
        self.share_WT_deg_types = share_WT_deg_types

        # In case data is provided as weeks per season
        self.weeks_per_season_per_year = weeks_per_season_per_year
        
    def setup(self):
        self.add_input('wst',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.N_time])
        self.add_input('wdt',
                       desc="wd time series at the hub height",
                       units='deg',
                       shape=[self.N_time])

        self.add_output('wst_ext',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.life_h])
        self.add_output('wdt_ext',
                       desc="wd time series at the hub height",
                       units='deg',
                       shape=[self.life_h])
        self.add_output('wind_t_ext_deg',
                        desc="power time series with degradation",
                        units='MW',
                        shape=[self.life_h])


    def compute(self, inputs, outputs):

        N_time = self.N_time
        life_h = self.life_h
        wpp_efficiency = self.wpp_efficiency

        existing_wpp_power_curve_xr = xr.open_dataset(self.existing_wpp_power_curve_xr_fn)
        
        # number of elements in WT degradation curve
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types

        # In case data is provided as weeks per season
        weeks_per_season_per_year = self.weeks_per_season_per_year
        
        wst = inputs['wst']
        wst_ext = expand_to_lifetime(
            wst, life_h = life_h, weeks_per_season_per_year = weeks_per_season_per_year)

        wdt = inputs['wdt']
        wdt_ext = expand_to_lifetime(
            wdt, life_h = life_h, weeks_per_season_per_year = weeks_per_season_per_year)

        xr_time = xr.Dataset()
        xr_time['wst'] = xr.DataArray( 
            data=wst_ext, 
            dims = ['t'],
            coords = {'t':np.arange(life_h)})
        xr_time['wdt'] = xr.DataArray( 
            data=wdt_ext, 
            dims = ['t'],
            coords = {'t':np.arange(life_h)})

        wake_losses_eff_t = existing_wpp_power_curve_xr.wake_losses_eff.interp(ws=xr_time.wst, wd=xr_time.wdt).values
        wake_losses_eff_t_ext = expand_to_lifetime(
            wake_losses_eff_t, life_h = life_h, weeks_per_season_per_year = weeks_per_season_per_year)

        ws = existing_wpp_power_curve_xr.ws.values
        pcw = existing_wpp_power_curve_xr.P_no_wake.values
        
        wind_t_ext_deg_no_wake = wpp_efficiency*get_wind_ts_degradation(
            ws = ws, 
            pc = pcw, 
            ws_ts = wst_ext, 
            yr = wind_deg_yr, 
            wind_deg = wind_deg, 
            life_h = life_h, 
            share = share_WT_deg_types)
        
        wind_t_ext_deg = wake_losses_eff_t_ext*wind_t_ext_deg_no_wake

        outputs['wst_ext'] = wst_ext
        outputs['wdt_ext'] = wdt_ext
        outputs['wind_t_ext_deg'] = wind_t_ext_deg


# -----------------------------------------------------------------------
# Auxiliar functions 
# -----------------------------------------------------------------------        

def get_rotor_area(d): return np.pi*(d/2)**2
def get_rotor_d(area): return 2*(area/np.pi)**0.5

def get_WT_curves(genWT_fn, specific_power):
    """
    Evaluates a generic WT look-up table

    Parameters
    ----------
    genWT_fn : look-up table filename
    specific_power : WT specific power

    Returns
    -------
    ws : Wind speed vector for power and thrust coefficient curves
    pc : Power curve
    ct : Thrust coefficient curves
    """
    genWT = xr.open_dataset(genWT_fn).interp(
        sp=specific_power, 
        kwargs={"fill_value": 0}
        )

    ws = genWT.ws.values
    pc = genWT.pc.values
    ct = genWT.ct.values
    
    genWT.close()
    
    return ws, pc, ct

def get_wake_affected_pc(
    genWake_fn, 
    specific_power,
    Nwt,
    wind_MW_per_km2,
    ws,
    pc,
    p_rated,
):
    """
    Evaluates a generic WT look-up table

    Parameters
    ----------
    genWake_fn : look-up table filename
    specific_power : WT specific power
    Nwt : Number of wind turbines
    wind_MW_per_km2 : Wind plant installation density
    ws : Wind speed vector for wake losses curves
    pc : 

    Returns
    -------
    wl : Wind plant wake losses curve
    """
    ds = xr.open_dataset(genWake_fn)
    ds_sel = ds.sel(Nwt=2)
    ds_sel['wl'] = 0*ds_sel['wl']
    ds_sel['Nwt'] = 1
    ds = xr.concat([ds_sel, ds], dim='Nwt')
    
    genWake_sm = ds.interp(
        ws=ws, 
        sp=float(specific_power), 
        Nwt=float(Nwt), 
        wind_MW_per_km2=float(wind_MW_per_km2),
        kwargs={"fill_value": 1}
        )
    wl = genWake_sm.wl.values
    
    genWake_sm.close()
    
    pcw = pc * (1 - wl)
    return pcw * Nwt * p_rated

def get_wind_ts(
    ws,
    pcw,
    wst,
    wpp_efficiency
):
    """
    Evaluates a generic WT look-up table

    Parameters
    ----------
    ws : Wind speed vector for wake losses curves
    pcw : Wake affected plant power curve
    wst : Wind speed time series

    Returns
    -------
    wind_ts : Wind plant power time series
    """
    wind_ts = wpp_efficiency * np.interp(wst, ws, pcw, left=0, right=0, period=None)
    return wind_ts


# ---------------------------------------------
# Auxiliar functions for wind plant degradation
# ---------------------------------------------
def get_prated_end(ws,pc,tol=1e-6):
    if np.max(pc)>0:
        pc = pc/np.max(pc)
        ind = np.where( (np.diff(pc)<=tol)&(pc[:-1]>=1-tol) )[0]
        ind_sel = [ind[0], ind[-1]]
        return ind[-1]
    return -3

def get_shifted_pc(ws,pc,Dws):
    ind_sel = get_prated_end(ws,pc)
    pcdeg_init = get_wind_ts(ws=ws+Dws, pcw=pc, wst=ws, wpp_efficiency=1)
    pcdeg = np.copy(pcdeg_init)
    pcdeg[ind_sel:] = pc[ind_sel:]
    return pcdeg

def get_losses_shift_power_curve(ws,pc,ws_ts,Dws):
    CF_ref = np.mean(get_wind_ts(ws=ws, pcw=pc, wst=ws_ts, wpp_efficiency=1))
    if CF_ref > 0:
        pcdeg = get_shifted_pc(ws,pc,Dws)
        CF_deg = np.mean(get_wind_ts(ws=ws, pcw=pcdeg, wst=ws_ts, wpp_efficiency=1))
        return (1-CF_deg/CF_ref)
    else:
        return np.NaN

def get_Dws(ws, pc, ws_ts, wind_deg_end):
    CF_ref = np.mean(get_wind_ts(ws=ws, pcw=pc, wst=ws_ts, wpp_efficiency=1))
    if CF_ref > 0:
        def fun(x, target):
            return (get_losses_shift_power_curve(ws,pc,ws_ts,Dws=x) - target)**2
    
        out = sp.optimize.minimize(
            fun=fun, 
            x0=0.5, 
            args=(wind_deg_end), 
            method='SLSQP',
            tol=1e-10)
    
        return out.x
    else:
        return 0.0
    
def get_wind_ts_degradation(ws, pc, ws_ts, yr, wind_deg, life_h, share=0.5):
    
    t_over_year = np.arange(life_h)/(365*24)
    #degradation = wind_deg_per_year * t_over_year
    degradation = np.interp(t_over_year, yr, wind_deg)

    p_ts = get_wind_ts(ws=ws, pcw=pc, wst=ws_ts, wpp_efficiency=1)
    Dws = get_Dws(ws, pc, ws_ts,wind_deg_end=degradation[-1])
    pcdeg = get_shifted_pc(ws,pc,Dws=Dws)
    p_ts_fulldeg = get_wind_ts(ws=ws, pcw=pcdeg, wst=ws_ts, wpp_efficiency=1)

    # blend variable for pc shift over time
    if np.max(wind_deg) <= 0:
        alpha = 0
    else:
        alpha = degradation/np.max(degradation)

    # degradation in CF as a results of a shift in ws on power curve
    p_ts_deg = (1-alpha)*p_ts + alpha*p_ts_fulldeg
    # degradation in CF as a factor or losses
    p_ts_deg_factor = (1-degradation)*p_ts

    p_ts_deg_partial_factor = (1-share)*p_ts_deg + share*p_ts_deg_factor

    # ws shift cannot handle large degradations. (degradation >0.8)
    p_ts_deg_partial_factor[np.where(degradation>0.8)[0]] = p_ts_deg_factor[np.where(degradation>0.8)[0]]
    
    return p_ts_deg_partial_factor

