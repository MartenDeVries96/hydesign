# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:43:04 2022

@author: mikf
"""
import numpy as np
# import matplotlib.pyplot as plt
from hydesign.Parallel_EGO import (
    get_sm, LCB, EI, KStd, KB, eval_sm, #opt_sm,
    get_candiate_points, get_mixint_context)
from hydesign.tests.test_files import tfp
import pandas as pd
import pytest
import pickle

def test_get_sm():
    nt = 100
    data = pd.read_csv(tfp + 'test_data.csv', sep=';',nrows=nt) # y=A2^2+B2^3+SIN(C2)+ATAN(D2)+1/E2+PI()*F2+EXP(G2/10)+H2^4+LN(I2)
    x = data[[f'x{int(i + 1)}' for i in range(9)]].values
    y = data['y2'].values.reshape(nt, 1)
    return get_sm(x, y)

def save_sm():
    sm = test_get_sm()
    with open(tfp + 'sm.pkl','wb') as f:
         pickle.dump(sm, f)

def load_sm():
    with open(tfp + 'sm.pkl','rb') as f:
        sm = pickle.load(f)
    return sm

sm = load_sm()
fmins = [1, 10, 50, 500, 10000]
point = np.array([10, 1, 0, 142, 2, 10, 0, 0, 1]).reshape((1, 9)) # analytical: 135.4796807

def get_data2(fmin):
    data2 = pd.read_csv(tfp + 'sm_pred_test_data.csv', sep=';').values
    n = fmins.index(fmin)
    a = data2[n * 5: n * 5 + 5, :9]
    b = data2[n * 5: n * 5 + 5, 9]
    return a, b

def generate_data2():
    new_data = np.zeros((25, 10))
    for n, fmin in enumerate(fmins):
        a, b = eval_sm(sm, mixint, npred=5, fmin=fmin)
        new_data[n*5:n*5+5,:9] = a
        new_data[n*5:n*5+5,9] = b.ravel()
    df = pd.DataFrame(new_data, columns=[f'x{i + 1}' for i in range(9)] + ['y'])
    df.to_csv(tfp + 'sm_pred_test_data.csv', sep=';', index=False)

def test_LCB():
    res = LCB(sm, point)
    np.testing.assert_allclose(res, np.array([[127.142855]]))

def test_EI():
    res = EI(sm, point)
    np.testing.assert_allclose(res, np.array([[-863.117345]]))

def test_KStd():
    res = KStd(sm, point)
    np.testing.assert_allclose(res, np.array([[3.2466]]))

def test_KB():
    res = KB(sm, point)
    np.testing.assert_allclose(res, np.array([[136.882655]]))

variables = {
    'a':
        {'var_type':'design',
         'limits':[4, 22],
         'types':'int'
         },
     'b':
        {'var_type':'design',
         'limits':[0.1, 4.5],
         'types':'float'
         },
    'c':
        {'var_type':'design',
         'limits':[-0.65, 1.75],
         'types':'float'
         },
    'd':
        {'var_type':'design',
         'limits':[8, 270],
         'types':'int'
         },
    'e':
        {'var_type':'design',
         'limits':[1, 4.9],
         'types':'float'
         },
    'f':
        {'var_type':'design',
         'limits':[-22, 43],
         'types':'int'
         },
    'g':
        {'var_type':'design',
         'limits':[-10, 9],
         'types':'float'
         },
     'h':
        {'var_type':'design',
         'limits': [-10, 9],
         'types':'float'
         },
     'i':
        {'var_type':'design',
         'limits':[1.1, 1.34],
         'types':'float'
         },
        }
mixint = get_mixint_context(variables, seed=0)

@pytest.mark.parametrize('fmin', fmins)
def test_eval_sm(fmin):
    generate_data2()
    a, b = eval_sm(sm, mixint, npred=5, fmin=fmin)
    a_ref, b_ref = get_data2(fmin)
    np.testing.assert_allclose(a, a_ref)
    np.testing.assert_allclose(b.ravel(), b_ref)

def test_get_candidate_points():
    xpred, ypred_LB = eval_sm(sm, mixint, npred=5, fmin=10)
    xnew = get_candiate_points(
    xpred, ypred_LB, 
    n_clusters = 1, 
    quantile = 1e-4) 
    np.testing.assert_allclose(xnew, np.array([[9, 4.06, 1.51, 86, 2.95, -3, -4.3, -0.5, 1.316]]))

    
