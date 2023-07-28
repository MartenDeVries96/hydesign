import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.getcwd(), 'msvr'))
from model.MSVR import MSVR
from model.utility import create_dataset, rmse
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, MinMaxScaler
import xarray as xr

class mvsr_predictor:
   '''
   Wrapper for the MSVR predictor (https://github.com/Analytics-for-Forecasting/msvr)
   '''

   def __init__(self, scaler_x, scaler_y, kernel='poly', degree=1, gamma=1):
      '''
      Initialize MSVR predictor

      Parameters
      ---------
      scaler_x : sklearn scalar to preprocess input data (the inputs to the predictor)
      scaler_y : sklearn scalar to preprocess output data (the outputs to be predicted)
      kernel : kernel for building the regressor. Used in sklearn's `pairwise_kernels`
           valid entries are:
            [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
      degree : degree of kernel if the kernel is specified as a polynomial
      gamma : argument to sklearn's `pairwise_kernels`
      '''
      self.scaler_x = scaler_x
      self.scaler_y = scaler_y
      self.msvr = MSVR(kernel=kernel, degree=degree, gamma=gamma) 

   def fit(self, train_x, train_y):
      '''
      Fit the MSVR predictor to a given set of data
 
      Parameters
      ---------
      train_x : input data (the inputs to the predictor)
      train_y : output data (the outputs to be predicted)
      '''
      self.scaler_x.fit(train_x)
      train_input = self.scaler_x.transform(train_x) 
      self.scaler_y.fit(train_y)
      train_target = self.scaler_y.transform(train_y)
      self.msvr.fit(train_input, train_target)

   def predict(self, test_x):
      '''
      Use the MSVR predictor to predict the responses associated with
      a set of inputs
 
      Parameters
      ---------
      test_x : input data (the inputs to the predictor)
      '''
      test_input = self.scaler_x.transform(test_x) 
      test_predict = self.msvr.predict(test_input)
      return self.scaler_y.inverse_transform(test_predict)

# This function accepts the xarray dataframe
#  it outputs 
def tall_vector(ds, list_vars=None):
    '''
    Helper function to re-organize data 
    '''
    
    # determine number of successful runs
    N_realize_nr = len(ds.realize_nr)
    #print('OK')
    #N_realize_nr = 200
    
    # option to only include a subset of the output variables
    if list_vars is None:
        list_vars = ds.data_vars
    
    # sizes is a dictionary describing the size of the output data associated with each output variable
    sizes = dict()

    # shapes is a dictionary describing the ...
    # question: how is this different than the sizes dictionary?
    shapes = dict()

    for ii, var in enumerate(list_vars):

        # record the shape of the data associated with each variable
        shapes[var] = ds[var].values.shape
        data = ds[var].values.reshape(-1, N_realize_nr)
        #data = ds[var].values.reshape([-1, N_realize_nr])
        sizes[var] = data.shape[0]
        if ii == 0:
            out = data
        else:
            out = np.vstack([ out, data] )

    return out, sizes, shapes


def loo_score(degree, i_loo=12):
   '''
   Validation helper function. 
   '''
   i_left = [ii for ii in range(X_all.shape[0]) if ii!=i_loo]
   X_train = X_all[i_left,:]
   Y_train = Y_all[i_left,:]
   X_test = X_all[[i_loo],:]
   Y_test = Y_all[[i_loo],:]

   model = mvsr_predictor(scaler_x=MinMaxScaler(), scaler_y=QuantileTransformer( n_quantiles=100, output_distribution='uniform'), degree=degree)
   model.fit(X_train, Y_train)
   Y_pred = model.predict(X_test)
   return np.sqrt(np.mean((Y_test - Y_pred) ** 2))

def total_score(degree):
   '''
   Validation helper function. 
   '''
   scores = []
   for i_loo in range(X_all.shape[0]):
      scores.append(loo_score(degree, i_loo=i_loo))
   return scores


outputLabelName = 'realization_of_inputs'

class bigModel:
   '''
   This is a class containing several models. Together, these act as a surrogate for the output of REALIZE. 
   Each model is trained to predict a specified output of a specified region during a specified year. 
   '''
    def __init__(self, datax, regions=None, outputs=None, years=None, scalerx=MinMaxScaler, scalerx_opts={}, scalery=QuantileTransformer, scalery_opts=dict(n_quantiles=100, output_distribution='uniform'), degree=1, gamma=1):
        '''
        Initialization of REALIZE surrogate class.

        Parameters
        ----------
        datax : xarray of data associated with parsed REALIZE samples
        regions : regions to include in surrogate. If `None` is passed, include all regions
        outputs : outputs to include in surrogate. If `None` is passed, include all outputs
        years : years to include in surrogate. If `None` is passed, include all years
        scalerx : scaler used to preprocess the input data
        scalery : the scaler used to preprocess the output data 
        scalerx_opts : dictionary of options to pass to the input scaler
        scalery_opts = dictionary of options to pass to the output scaler
        degree : kernel polynomial degree
        gamma : sklearn kernel parameter
        '''
        # isolate inputs (they only depend on realize_nr)
        inputs_df = datax[[var for var in datax.data_vars if set(datax[var].dims) == {'realize_nr'}]].to_dataframe()

        # record the input names
        self.input_cols = inputs_df.columns
        self.weathertime = datax.weather_time.values
        X_train = inputs_df.values
        #X_train = inputs_df.iloc[:,1:].values
        
        if regions: self.regions = np.array(regions)
        else: self.regions = datax.region
            
        if outputs: self.outputs = np.array(outputs)
        else: self.outputs = np.array([var for var in datax.data_vars if set(datax[var].dims) != {'realize_nr'}])
        
        if years: self.years = years
        else: self.years = datax.year.values
            
        self.cross_regions = datax.region.values
            
        self.output_sizes = {output:datax[output].shape[1] for output in self.outputs}
            
        self.models = {}
        to_save = {}
        for output in self.outputs:
            for yy, year in enumerate(self.years):
                for rr, region in enumerate(self.regions):
                    model = mvsr_predictor(scaler_x=scalerx(**scalerx_opts), scaler_y=scalery(**scalery_opts), degree=degree, gamma=gamma)
                    ds_sel = datax.sel(year=year, region=region)
                    out, sizes, shapes = tall_vector(ds=ds_sel, list_vars=[output])
                    Y_train=out.T
                    model.fit(X_train, Y_train)
                    self.models[(output, yy, rr)] = model

    def predict(self, new_scenarios, scenario_numbers=None, output=None):
        '''
        Predict outputs based in unseen inputs using the MSVR models

        Parameters
        ----------
        new_scenarios : numpy array or pandas dataframe of new input prices to evaluate.
                        If numpy array, the columns of the matrix should be ordered according to self.input_cols
                        If pandas array, the columns should be named using the names in self.input_cols
        scenario_numbers : the numbers to associate with each prediction output by the model. 
                           Each number corresponds to a row of input variables. Default in 1:N-1
        output : output to be predicted. If None, predict all outputs the model has been trained on.
        '''
        n_scenarios = new_scenarios.shape[0]
        if type(new_scenarios) == pd.DataFrame and scenario_numbers is None:
          scenario_numbers = new_scenarios.index.values
        if type(new_scenarios) in [list, np.ndarray]:
            new_scenarios = pd.DataFrame(new_scenarios, columns=self.input_cols)
        x_in = new_scenarios[self.input_cols].values
        if 'realize_nr' in new_scenarios.columns:
           scenario_nums = new_scenarios['realize_nr']
        elif scenario_numbers is not None: scenario_nums=scenario_numbers
        else:
           scenario_nums = range(n_scenarios)
        if output: outputs_to_predict = output
        else: outputs_to_predict = self.outputs
        
        to_save = {}
        for oo, output in enumerate(outputs_to_predict):
            if output in ['VRE_Capacities_Wind_Onshore', 'VRE_Capacities_Wind_Offshore', 'VRE_Capacities_Solar_PV']:
                 saver = np.zeros((len(self.years), self.regions.size, n_scenarios))
            elif output in ['Transmission_Lines_Capacities']:
                 saver = np.zeros((len(self.years), self.regions.size, self.regions.size, n_scenarios, ))
            else:
                 saver = np.zeros((len(self.years), self.regions.size, self.output_sizes[output], n_scenarios, ))
            for yy, year in enumerate(self.years):
                for rr, region in enumerate(self.regions):
                    saver[yy, rr, :] = self.models[outputs_to_predict[oo], yy, rr].predict(x_in).T
                    #saver[yy, rr, oo] = np.split(self.models[yy, rr].predict(x_in).T, len(self.outputs))[oo]
            if output in ['VRE_Capacities_Wind_Onshore', 'VRE_Capacities_Wind_Offshore', 'VRE_Capacities_Solar_PV']:
                to_save[output] = xr.DataArray(saver, dims=['year', 'region', outputLabelName])
            elif output in ['Transmission_Lines_Capacities']:
                to_save[output] = xr.DataArray(saver, dims=['year', 'region', 'cross_region', outputLabelName])
            else:
                to_save[output] = xr.DataArray(saver, dims=['year', 'region', 'weathertime', outputLabelName])

        # save input
        for col in self.input_cols:
           to_save[col] = xr.DataArray(new_scenarios[col].values, dims=[outputLabelName])
           #to_save[col] = new_scenarios[col]

        return xr.Dataset(data_vars=to_save, coords={'year': self.years, 'weathertime': self.weathertime, 'region': self.regions, 'cross_region': self.cross_regions, outputLabelName: scenario_nums}).transpose("year", "weathertime", "region", outputLabelName, 'cross_region')
            
