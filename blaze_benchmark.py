# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:50:05 2015

@author: bolaka
"""

# suppress pandas warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

# imports
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import preprocessing
from numpy.random import seed
from blaze import CSV, Table, concat

# reproduce results
seed(786)

train_csv = CSV('train.csv')
train = Table(train_csv)

test_csv = CSV('test.csv')
test = Table(test_csv)

combined = concat(train, test, axis=1)
combined.dshape