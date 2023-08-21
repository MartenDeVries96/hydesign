# %% finance model with P2X with hydrogen storage and transportation
import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd
# import seaborn as sns
import openmdao.api as om
import yaml

class finance(om.ExplicitComponent):
    """Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant.
    It considers different weighted average costs of capital (WACC) for wind, PV and battery. The model calculates
    the yearly cashflow as a function of the average revenue over the year, the tax rate and WACC after tax
    ( = weighted sum of the wind, solar, battery, and electrical infrastracture WACC). Net present value (NPV)
    and levelized cost of energy (LCOE) is then be calculated using the calculates WACC as the discount rate, as well
    as the internal rate of return (IRR).
    """

    def __init__(self, 
                 N_time, 
                 life_h = 25*365*24,
                ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """ 
        super().__init__()
        self.N_time = int(N_time)
        self.life_h = int(life_h)

    def setup(self):
        self.add_input('price_t_ext',
                       desc="Electricity price time series",
                       shape=[self.life_h])
        
        self.add_input('hpp_t_with_deg',
                       desc="HPP power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('penalty_t',
                        desc="penalty for not reaching expected energy productin at peak hours",
                        shape=[self.life_h])

        self.add_input('CAPEX_w',
                       desc="CAPEX wpp")
        self.add_input('OPEX_w',
                       desc="OPEX wpp")

        self.add_input('CAPEX_s',
                       desc="CAPEX solar pvp")
        self.add_input('OPEX_s',
                       desc="OPEX solar pvp")

        self.add_input('CAPEX_b',
                       desc="CAPEX battery")
        self.add_input('OPEX_b',
                       desc="OPEX battery")

        self.add_input('CAPEX_el',
                       desc="CAPEX electrical infrastructure")
        self.add_input('OPEX_el',
                       desc="OPEX electrical infrastructure")

        self.add_input('wind_WACC',
                       desc="After tax WACC for onshore WT")
        
        self.add_input('solar_WACC',
                       desc="After tax WACC for solar PV")
        
        self.add_input('battery_WACC',
                       desc="After tax WACC for stationary storge li-ion batteries")
        
        self.add_input('tax_rate',
                       desc="Corporate tax rate")
        
        self.add_output('CAPEX',
                        desc="CAPEX")
        
        self.add_output('OPEX',
                        desc="OPEX")
        
        self.add_output('NPV',
                        desc="NPV")
        
        self.add_output('IRR',
                        desc="IRR")
        
        self.add_output('NPV_over_CAPEX',
                        desc="NPV/CAPEX")
        
        self.add_output('mean_AEP',
                        desc="mean AEP")
        
        self.add_output('LCOE',
                        desc="LCOE")
        
        self.add_output('penalty_lifetime',
                        desc="penalty_lifetime")

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """ Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        price_t_ext : Electricity price time series [Eur]
        hpp_t_with_deg : HPP power time series [MW]
        penalty_t : penalty for not reaching expected energy productin at peak hours [Eur]
        CAPEX_w : CAPEX of the wind power plant
        OPEX_w : OPEX of the wind power plant
        CAPEX_s : CAPEX of the solar power plant
        OPEX_s : OPEX of solar power plant   
        CAPEX_b : CAPEX of the battery
        OPEX_b : OPEX of the battery
        CAPEX_sh :  CAPEX of the shared electrical infrastracture
        OPEX_sh : OPEX of the shared electrical infrastracture
        wind_WACC : After tax WACC for onshore WT
        solar_WACC : After tax WACC for solar PV
        battery_WACC: After tax WACC for stationary storge li-ion batteries
        tax_rate : Corporate tax rate

        Returns
        -------
        CAPEX : Total capital expenditure costs of the HPP
        OPEX : Operational and maintenance costs of the HPP
        NPV : Net present value
        IRR : Internal rate of return
        NPV_over_CAPEX : NPV over CAPEX
        mean_AEP : Mean annual energy production
        LCOE : Levelized cost of energy
        penalty_lifetime : total penalty
        """
        
        N_time = self.N_time
        life_h = self.life_h
        
        df = pd.DataFrame()
        
        df['hpp_t'] = inputs['hpp_t_with_deg']
        df['price_t'] = inputs['price_t_ext']
        df['penalty_t'] = inputs['penalty_t']
        df['revenue'] = df['hpp_t'] * df['price_t'] - df['penalty_t']
        
        df['i_year'] = np.hstack([np.array([ii]*N_time) 
                                  for ii in range(int(np.ceil(life_h/N_time)))])[:life_h]

        revenues = df.groupby('i_year').revenue.mean()*365*24
        CAPEX = inputs['CAPEX_w'] + inputs['CAPEX_s'] + \
            inputs['CAPEX_b'] + inputs['CAPEX_el']
        OPEX = inputs['OPEX_w'] + inputs['OPEX_s'] + \
            inputs['OPEX_b'] + inputs['OPEX_el']
        
        outputs['CAPEX'] = CAPEX
        outputs['OPEX'] = OPEX
        
        # len of revenues
        iy = np.arange(len(revenues)) + 1
        Net_revenue_t = revenues.values.flatten()
        
        WACC_after_tax = calculate_WACC(
            inputs['CAPEX_w'],
            inputs['CAPEX_s'],
            inputs['CAPEX_b'],
            inputs['CAPEX_el'],
            inputs['wind_WACC'],
            inputs['solar_WACC'],
            inputs['battery_WACC'],
            )
        
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t = revenues.values.flatten(),
            investment_cost = CAPEX,
            maintenance_cost_per_year = OPEX,
            tax_rate = inputs['tax_rate'],
            WACC_after_tax = WACC_after_tax)
        
        hpp_discount_factor = WACC_after_tax

        outputs['NPV'] = NPV
        outputs['IRR'] = IRR
        outputs['NPV_over_CAPEX'] = NPV / CAPEX

        level_costs = np.sum(OPEX / (1 + hpp_discount_factor)**iy) + CAPEX
        AEP_per_year = df.groupby('i_year').hpp_t.mean()*365*24
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor)**iy)

        mean_AEP_per_year = np.mean(AEP_per_year)
        if level_AEP > 0:
            outputs['LCOE'] = level_costs / (level_AEP) # in Euro/MWh
        else:
            outputs['LCOE'] = 1e6

        outputs['mean_AEP'] = mean_AEP_per_year
        
        outputs['penalty_lifetime'] = df['penalty_t'].sum()


class finance_P2X(om.ExplicitComponent):
    """Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant with P2X.
    It considers different weighted average costs of capital (WACC) for wind, PV, battery and P2X. The model calculates
    the yearly cashflow as a function of the average revenue over the year, the tax rate and WACC after tax
    ( = weighted sum of the wind, solar, battery, P2X and electrical infrastracture WACC). Net present value (NPV)
    and levelized cost of energy (LCOE) is then be calculated using the calculates WACC as the discount rate, as well
    as the internal rate of return (IRR).
    """

    def __init__(self, 
                 N_time, 
                 life_h = 25*365*24,
                ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """ 
        super().__init__()
        self.N_time = int(N_time)
        self.life_h = int(life_h)

    def setup(self):
        self.add_input('price_t_ext',
                       desc="Electricity price time series",
                       shape=[self.life_h])
        
        self.add_input('hpp_t',
                       desc="HPP power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('penalty_t',
                        desc="penalty for not reaching expected energy productin at peak hours",
                        shape=[self.life_h])
        
        self.add_input('m_H2_t',
                       desc = "Produced Hydrogen",
                       units = 'kg',
                       shape=[self.life_h])

        self.add_input('m_H2_offtake_t',
                       desc = "Produced Hydrogen",
                       units = 'kg',
                       shape=[self.life_h])
        
        self.add_input('m_H2_demand_t_ext',
                       desc = "Hydrogen demand times series",
                       units = 'kg',
                       shape=[self.life_h])
        
        self.add_input('P_ptg_t',
                       desc = "Electrolyzer power consumption time series",
                       units = 'MW',
                       shape=[self.life_h])
        
        self.add_input('price_H2',
                       desc = "H2 price")
        
        self.add_input('CAPEX_w',
                       desc="CAPEX wpp")
        self.add_input('OPEX_w',
                       desc="OPEX wpp")

        self.add_input('CAPEX_s',
                       desc="CAPEX solar pvp")
        self.add_input('OPEX_s',
                       desc="OPEX solar pvp")

        self.add_input('CAPEX_b',
                       desc="CAPEX battery")
        self.add_input('OPEX_b',
                       desc="OPEX battery")

        self.add_input('CAPEX_el',
                       desc="CAPEX electrical infrastructure")
        self.add_input('OPEX_el',
                       desc="OPEX electrical infrastructure")
        
        self.add_input('CAPEX_ptg',
                       desc = "CAPEX ptg plant")
        self.add_input("OPEX_ptg",
                       desc = "OPEX ptg plant")
        self.add_input("water_consumption_cost",
                       desc = "Water usage and purification for the electrolysis")

        self.add_input('wind_WACC',
                       desc="After tax WACC for onshore WT")
        
        self.add_input('solar_WACC',
                       desc="After tax WACC for solar PV")
        
        self.add_input('battery_WACC',
                       desc="After tax WACC for stationary storge li-ion batteries")
        
        self.add_input('ptg_WACC',
                       desc = "After tax WACC for power to gas plant")
        
        self.add_input('tax_rate',
                       desc="Corporate tax rate")
        
        self.add_output('CAPEX',
                        desc="CAPEX")
        
        self.add_output('OPEX',
                        desc="OPEX")
        
        self.add_output('NPV',
                        desc="NPV")
        
        self.add_output('IRR',
                        desc="IRR")
        
        self.add_output('NPV_over_CAPEX',
                        desc="NPV/CAPEX")
        
        self.add_output('mean_AEP',
                        desc="mean AEP")
        
        self.add_output('annual_H2',
                        desc = "Annual H2 production")
        
        self.add_output('LCOE',
                        desc="LCOE")
        
        self.add_output('penalty_lifetime',
                        desc="penalty_lifetime")

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """ Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        price_t_ext : Electricity price time series [Eur]
        hpp_t_with_deg : HPP power time series [MW]
        penalty_t : penalty for not reaching expected energy productin at peak hours [Eur]
        m_H2_t: Produced Hydrogen
        m_H2_offtake_t: Hydrogen offtake time series
        m_H2_demand_t_ext: Hydrogen demand times series
        P_ptg_t: Electrolyzer power consumption time series
        price_H2: H2 price
        CAPEX_w : CAPEX of the wind power plant
        OPEX_w : OPEX of the wind power plant
        CAPEX_s : CAPEX of the solar power plant
        OPEX_s : OPEX of solar power plant   
        CAPEX_b : CAPEX of the battery
        OPEX_b : OPEX of the battery
        CAPEX_ptg : CAPEX of P2G plant
        OPEX_ptg : OPEX of P2G plant
        CAPEX_sh :  CAPEX of the shared electrical infrastracture
        OPEX_sh : OPEX of the shared electrical infrastracture
        wind_WACC : After tax WACC for onshore WT
        solar_WACC : After tax WACC for solar PV
        battery_WACC: After tax WACC for stationary storge li-ion batteries
        ptg_WACC: After tax WACC for power to gas plant
        tax_rate : Corporate tax rate

        Returns
        -------
        CAPEX : Total capital expenditure costs of the HPP
        OPEX : Operational and maintenance costs of the HPP
        NPV : Net present value
        IRR : Internal rate of return
        NPV_over_CAPEX : NPV over CAPEX
        mean_AEP : Mean annual energy production
        annual_H2: Annual H2 production
        LCOE : Levelized cost of energy
        penalty_lifetime : total penalty
        """
        
        N_time = self.N_time
        life_h = self.life_h
        
        df = pd.DataFrame()
        
        df['hpp_t'] = inputs['hpp_t']
        df['m_H2_t'] = inputs['m_H2_t']
        df['m_H2_offtake_t'] = inputs['m_H2_offtake_t']
        df['P_ptg_t'] = inputs['P_ptg_t']
        price_H2 = inputs['price_H2']
        df['price_t'] = inputs['price_t_ext']
        df['penalty_t'] = inputs['penalty_t']
        df['m_H2_demand_t_ext'] = inputs['m_H2_demand_t_ext']
        df['revenue'] = df['hpp_t'] * df['price_t'] + df['m_H2_offtake_t'] * price_H2 - 0.5 * (df['m_H2_demand_t_ext'] - df['m_H2_offtake_t']) - df['penalty_t']
        
        df['i_year'] = np.hstack([np.array([ii]*N_time) 
                                  for ii in range(int(np.ceil(life_h/N_time)))])[:life_h]

        revenues = df.groupby('i_year').revenue.mean()*365*24
        CAPEX = inputs['CAPEX_w'] + inputs['CAPEX_s'] + \
            inputs['CAPEX_b'] + inputs['CAPEX_el'] + inputs['CAPEX_ptg']
        OPEX = inputs['OPEX_w'] + inputs['OPEX_s'] + \
            inputs['OPEX_b'] + inputs['OPEX_el'] + inputs['OPEX_ptg'] + inputs['water_consumption_cost']
        
        outputs['CAPEX'] = CAPEX
        outputs['OPEX'] = OPEX
        
        # len of revenues
        iy = np.arange(len(revenues)) + 1
        Net_revenue_t = revenues.values.flatten()
        
        WACC_after_tax = calculate_WACC_P2X(
            inputs['CAPEX_w'],
            inputs['CAPEX_s'],
            inputs['CAPEX_b'],
            inputs['CAPEX_el'],
            inputs['CAPEX_ptg'],
            inputs['wind_WACC'],
            inputs['solar_WACC'],
            inputs['battery_WACC'],
            inputs['ptg_WACC'],
            )
        
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t = revenues.values.flatten(),
            investment_cost = CAPEX,
            maintenance_cost_per_year = OPEX,
            tax_rate = inputs['tax_rate'],
            WACC_after_tax = WACC_after_tax)
        
        hpp_discount_factor = WACC_after_tax

        outputs['NPV'] = NPV
        outputs['IRR'] = IRR
        outputs['NPV_over_CAPEX'] = NPV / CAPEX

        level_costs = np.sum(OPEX / (1 + hpp_discount_factor)**iy) + CAPEX
        AEP_per_year = df.groupby('i_year').hpp_t.mean()*365*24 + df.groupby('i_year').P_ptg_t.mean()*365*24
        annual_H2 = df.groupby('i_year').m_H2_t.mean()*365*24
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor)**iy)

        mean_AEP_per_year = np.mean(AEP_per_year)
        annual_H2 = np.mean(annual_H2)
        if level_AEP > 0:
            outputs['LCOE'] = level_costs / (level_AEP) # in Euro/MWh
        else:
            outputs['LCOE'] = 1e6

        outputs['mean_AEP'] = mean_AEP_per_year
        outputs['annual_H2'] = annual_H2
        outputs['penalty_lifetime'] = df['penalty_t'].sum()

# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------


def calculate_NPV_IRR(
    Net_revenue_t,
    investment_cost,
    maintenance_cost_per_year,
    tax_rate,
    WACC_after_tax):
    """ A function to estimate the yearly cashflow using the net revenue time series, and the yearly OPEX costs.
    It then calculates the NPV and IRR using the yearly cashlow, the CAPEX, the WACC after tax, and the tax rate.

    Parameters
    ----------
    Net_revenue_t : Net revenue time series
    investment_cost : Capital costs
    maintenance_cost_per_year : yearly operation and maintenance costs
    tax_rate : tax rate
    WACC_after_tax : Weighted average cost of capital after tax

    Returns
    -------
    NPV : Net present value
    IRR : Internal rate of return
    """


    # EBIT: earnings before interest and taxes
    EBIT = (Net_revenue_t - maintenance_cost_per_year) 
    
    # WACC: weighted average cost of capital
    Net_income = (EBIT*(1-tax_rate))*(1-WACC_after_tax) 
    Cashflow = np.insert(Net_income, 0, -investment_cost)
    NPV = npf.npv(WACC_after_tax, Cashflow)
    if NPV > 0:
        IRR = npf.irr(Cashflow)    
    else:
        IRR = 0
    return NPV, IRR

def calculate_WACC(
    CAPEX_w,
    CAPEX_s,
    CAPEX_b,
    CAPEX_el,
    wind_WACC,
    solar_WACC,
    battery_WACC,
    ):
    """ This function returns the weighted average cost of capital after tax, using solar, wind, and battery
    WACC. First the shared costs WACC is computed by taking the mean of the WACCs across all technologies.
    Then the WACC after tax is calculated by taking the weighted sum by the corresponding CAPEX.

    Parameters
    ----------
    CAPEX_w : CAPEX of the wind power plant
    CAPEX_s : CAPEX of the solar power plant
    CAPEX_b : CAPEX of the battery
    CAPEX_el : CAPEX of the shared electrical costs
    wind_WACC : After tax WACC for onshore WT
    solar_WACC : After tax WACC for solar PV
    battery_WACC : After tax WACC for stationary storge li-ion batteries

    Returns
    -------
    WACC_after_tax : WACC after tax
    """

    # Weighted average cost of capital 
    WACC_after_tax = \
        ( CAPEX_w * wind_WACC + \
          CAPEX_s * solar_WACC + \
          CAPEX_b * battery_WACC + \
          CAPEX_el * (wind_WACC + solar_WACC + battery_WACC)/3 ) / \
        ( CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_el )
    return WACC_after_tax


def calculate_WACC_P2X(
    CAPEX_w,
    CAPEX_s,
    CAPEX_b,
    CAPEX_el,
    CAPEX_ptg,
    wind_WACC,
    solar_WACC,
    battery_WACC,
    ptg_WACC,
    ):
    """ This function returns the weighted average cost of capital after tax, using solar, wind, electrolyzer and battery
    WACC. First the shared costs WACC is computed by taking the mean of the WACCs across all technologies.
    Then the WACC after tax is calculated by taking the weighted sum by the corresponding CAPEX.

    Parameters
    ----------
    CAPEX_w : CAPEX of the wind power plant
    CAPEX_s : CAPEX of the solar power plant
    CAPEX_b : CAPEX of the battery
    CAPEX_el : CAPEX of the shared electrical costs
    wind_WACC : After tax WACC for onshore WT
    solar_WACC : After tax WACC for solar PV
    battery_WACC : After tax WACC for stationary storge li-ion batteries
    ptg_WACC : After tax WACC for power to gas plant

    Returns
    -------
    WACC_after_tax : WACC after tax
    """

    # Weighted average cost of capital 
    WACC_after_tax = \
        ( CAPEX_w * wind_WACC + \
          CAPEX_s * solar_WACC + \
          CAPEX_b * battery_WACC + \
          CAPEX_ptg * ptg_WACC +\
          CAPEX_el * (wind_WACC + solar_WACC + battery_WACC + ptg_WACC)/4 ) / \
        ( CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_el + CAPEX_ptg)
    return WACC_after_tax
