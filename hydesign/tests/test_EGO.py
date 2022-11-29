# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:43:04 2022

@author: mikf
"""
import numpy as np
import matplotlib.pyplot as plt
from hydesign.EGO_surrogate_based_optimization import get_sm, LCB
from hydesign.tests.test_files import tfp
import pandas as pd

nt = 100
data = pd.read_csv(tfp + 'test_data.csv', sep=';',nrows=nt)
x = data[[f'x{int(i + 1)}' for i in range(9)]].values
y = data['y'].values.reshape(nt, 1)
sm = get_sm(x, y)

def test_LCB():
    point = np.array([10, 1, 0, 142, 2, 10, 0, 0, 1]).reshape((1, 9))
    res = LCB(sm, point)
    np.testing.assert_allclose(res, np.array([[-52174.21032693]]))