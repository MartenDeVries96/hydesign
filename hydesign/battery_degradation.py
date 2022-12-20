# %%

import glob
import os
import time
import copy

# basic libraries
import numpy as np
from numpy import newaxis as na
import pandas as pd
import openmdao.api as om
import yaml

import xarray as xr
from docplex.mp.model import Model

import rainflow

class battery_degradation(om.ExplicitComponent):
    """
    Battery degradation model to predict the degradation of the battery throughout the lifetime of the plant

    Parameters
    ----------
    b_E_SOC_t : battery energy SOC time series
    min_LoH : minimum level of health before death of battery

    Returns
    -------
    ii_time : indices on the lifetime timeseries on which Hydesign operates in at constant battery health
    SoH : battery state of health at discretization levels
    """

    def __init__(
        self, 
        num_batteries = 1,
        n_steps_in_LoH = 30,
        life_h = 25*365*24):

        super().__init__()
        self.life_h = life_h
        self.num_batteries = num_batteries
        self.n_steps_in_LoH = n_steps_in_LoH

    def setup(self):
        self.add_input(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])
        self.add_input(
            'min_LoH',
            desc="minimum level of health before death of battery")
        
        # -------------------------------------------------------

        self.add_output(
            'ii_time',
            desc="indices on the lifetime timeseries on which"+
                " Hydesign operates in at constant battery health",
            shape=[self.n_steps_in_LoH*self.num_batteries + 1 ])
        self.add_output(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.n_steps_in_LoH*self.num_batteries + 1])

    def compute(self, inputs, outputs):

        num_batteries = self.num_batteries
        n_steps_in_LoH = self.n_steps_in_LoH

        b_E_SOC_t = inputs['b_E_SOC_t']
        min_LoH = inputs['min_LoH'][0]
        
        try:
            if np.max(b_E_SOC_t) == 0:
                nn = self.n_steps_in_LoH*self.num_batteries + 1
                outputs['ii_time'] = np.linspace(0,self.life_h, nn, dtype=int, endpoint=False)
                outputs['SoH'] = np.ones(nn)
            else:
                SoC = b_E_SOC_t/np.max(b_E_SOC_t)
                rf_DoD, rf_SoC, rf_count, rf_i_start = RFcount(SoC)   

                # To do: use the temperature time-series
                avr_tem = 20

                LoC, ind_q, _ = battery_replacement(
                    rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, 
                    min_LoH, n_steps_in_LoH, num_batteries)

                outputs['ii_time'] = rf_i_start[ind_q].astype(int)
                outputs['SoH'] = 1 - LoC[ind_q]
            
        except:
            nn = self.n_steps_in_LoH*self.num_batteries + 1
            outputs['ii_time'] = np.linspace(0,self.life_h, nn, dtype=int, endpoint=False)
            outputs['SoH'] = np.ones(nn)
        
        
            

# -----------------------------------------------------------------------
# Auxiliar functions for bat_deg modelling
# -----------------------------------------------------------------------


def battery_replacement(
    rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, 
    min_LoH, n_steps_in_LoH, num_batteries):
    """
    Battery degradation in steps and battery replacement

    Parameters
    ----------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    avr_tem: average temperature in the location, yearly or more long. default value is 20
    min_LoH: minimum level of health before death of battery
    n_steps_in_LoH: number of discretizations in battery state of health
    num_batteries: number of battery replacements

    Returns
    -------
    LoC: battery level of capacity
    ind_q: time indices for constant health levels
    ind_q_last: time index for battery replacement
    """

    #rf_DoD: depth of discharge after rainflow counting
    #rf_SoC: mean SoC after rainflow counting
    #rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    #rf_i_start: time index for the cycles [in hours]
    #avr_tem: average temperature in the location, yearly or more long. default value is 20

    #LoC: loss of capacity: LoC = 1 - LoH 
    
    LoC, LoC1, LLoC  = degradation(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, LLoC_0=0)

    ind_q = [np.where(1-LoC < q)[0][0] 
             for q in np.linspace(1,min_LoH,n_steps_in_LoH+1, endpoint = True)]
    ind_q_last = ind_q[-1]

    LoC[ind_q_last:] = 1
    
    # Battery replacement
    for i in range(num_batteries-1):
        
        try:
            # Degradation is computed after the new battery is installed: ind_q_last
            LoC_new, LoC1_new, LLoC_new  = degradation(
                rf_DoD[ind_q_last:], 
                rf_SoC[ind_q_last:], 
                rf_count[ind_q_last:], 
                rf_i_start[ind_q_last:]-rf_i_start[ind_q_last], 
                avr_tem, 
                LLoC_0=0, # starts with new battery without degradation
            )

            LoC[ind_q_last:] = LoC_new

            if min_LoH >  (1 - LoC_new[-1]):
                ind_q_new = [np.where(1-LoC_new < q)[0][0] + ind_q_last
                             for q in np.linspace(1,min_LoH,n_steps_in_LoH+1, endpoint = True)]
                ind_q_last = ind_q_new[-1] 
                ind_q = ind_q + ind_q_new[1:]

                LoC[ind_q_last:] = 1
            else:
                ind_q_new = [np.where(1-LoC_new < q)[0][0] + ind_q_last
                             for q in np.linspace(1,1-LoC_new[-1],n_steps_in_LoH+1, endpoint = False)]
                ind_q_last = ind_q_new[-1] 
                ind_q = ind_q + ind_q_new[1:]        
        except:
            raise('This many bateries are not required. Reduce the number.')
            

    return LoC, ind_q, ind_q_last

def degradation(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, LLoC_0=0):
    """
    Calculating the new level of capacity of the battery

    Parameters
    ----------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    avr_tem: average temperature in the location, yearly or more long. default value is 20

    Returns
    -------
    LoC: battery level of capacity
    LoC1: 
    LLoC: 
    """
    #rf_DoD: depth of discharge after rainflow counting
    #rf_SoC: mean SoC after rainflow counting
    #rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    #rf_i_start: time index for the cycles [in hours]
    #avr_tem: average temperature in the location, yearly or more long. default value is 20

    #SoH: state of health = 1 - loss of capacity, between 0 and 1
    #LoC: loss of capacity
    #LLoC: linear estimation of LoC 

    LLoC_hist = Linear_Degfun(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem)
    
    alpha = 0.0575
    beta = 121
    LLoC = LLoC_0 + np.cumsum(LLoC_hist)
    LLoC1 = LLoC.copy()
    LoC1 = 1-alpha*np.exp(-LLoC*beta)-(1-alpha)*np.exp(-LLoC)
    
    SoH_l = 1-LoC1
    ind_SoH_lt_92 = np.where(SoH_l<=0.92)[0]
    LoC = LoC1.copy()
    LoC[ind_SoH_lt_92] = 1-(1-LoC1[ind_SoH_lt_92])*np.exp(-(LLoC[ind_SoH_lt_92]-LoC1[ind_SoH_lt_92]))
    LoC[ind_SoH_lt_92] = LoC[ind_SoH_lt_92] + LoC1[ind_SoH_lt_92[0]] - LoC[ind_SoH_lt_92[0]]
    
    return LoC, LoC1, LLoC 

def Linear_Degfun(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem): 
    """
    Linear degradation function

    Parameters
    ----------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    avr_tem: average temperature in the location, yearly or more long. default value is 20

    Returns
    -------
    np.array(LLoC_hist): 

    """
    #LLoC:linear estimation of LoC
    #S_DoD:stress model of depth of discharge
    #S_time:stress model of time duration
    #S_SoC: stress model of state of charge
    #S_T: stress model of cell temperature
        
    kdelta1 = 140000
    kdelta2 = -0.5010
    kdelta3 = -123000
    ksigma = 1.04
    sigma_ref = 0.5
    kT = 0.0693
    Tref = 25
    kti = 4.14e-10
          
    LLoC_hist = []
    for j in range(len(rf_DoD)):
        #S_DoD = (kdelta1*rf_DoD[j]**kdelta2+kdelta3)**(-1)*rf_count[j]*2
        
        # To ensure no divide by zero problems
        aux = rf_DoD[j]
        if aux != 0:
            term = aux**kdelta2
        else:
            aux += 1e-12
            term = aux**kdelta2        
        S_DoD = (kdelta1*term+kdelta3)**(-1)*rf_count[j]*2
        
        #S_time = kti*(age_day*24/sum(rf_count)*rf_count[j]*3600)
        S_time = kti* rf_i_start[j]
        S_SoC = np.exp(ksigma*(rf_SoC[j]-sigma_ref))
        S_T = np.exp(kT*(avr_tem-Tref)*Tref/avr_tem)
                                
        LLoC_i = (S_DoD+S_time)*S_SoC*S_T
        LLoC_hist += [LLoC_i]
                
                            
    return np.array(LLoC_hist)
        
        
def RFcount(SoC):
    rf_df = pd.DataFrame(
        data=np.array([[rng, mean, count, i_start, i_end]  
                       for rng, mean, count, i_start, i_end  in rainflow.extract_cycles(SoC)]),
        columns=['rng_', 'mean_', 'count_', 'i_start', 'i_end']
    )
    """
    Rainflow count

    Parameters
    ----------
    SoC : state of charge time series

    Returns
    -------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    """
    
    rf_df = rf_df.sort_values(by='i_start')
    return rf_df.rng_.values, rf_df.mean_.values, rf_df.count_.values, rf_df.i_start.astype(int).values



    