# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:40:53 2015

@author: bolaka
"""

# suppress pandas warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

# imports
#import os
import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
#import numpy as np
from sklearn import preprocessing
#from sklearn.metrics import roc_curve, auc
#from ggplot import *
from numpy.random import seed
#from scipy.special import cbrt
#import matplotlib.pyplot as plt
#from sklearn.cross_validation import KFold
#from sklearn.cross_validation import StratifiedKFold

# reproduce results
seed(786)

# sum uo Missing
def sumMissing(s):
    return (s.isnull().sum()/len(s.isnull()))*100

# load the training and test sets
#tp = pd.read_csv('~/datasets/Springleaf/train.csv', iterator=True, chunksize=1000)
data = pd.read_csv('~/datasets/Springleaf/train.csv', nrows = 20000)
#data = pd.concat(tp, ignore_index=True) # df is DataFrame. If error do list(tp)

test_tp=pd.read_csv('~/datasets/Springleaf/test.csv', iterator=True, chunksize=5000)
test = pd.concat(test_tp, ignore_index=True)

test['target'] = 9999

# sum up missing    
missingness = data.apply(sumMissing, 0)    
sparseCols = missingness[missingness>50].index.values
print('dropping', len(sparseCols), 'columns')

# remove the target columns from data
data.drop(sparseCols, axis=1, inplace=True)   

categorical = list(data.select_dtypes(include=['object']).columns)

for cat in categorical:
    data[cat] = pd.Categorical.from_array(data[cat]).labels

data.fillna(-1,inplace=True)

features = list(data.columns)
features.remove('target')
x = data[features].values
y = data['target'].values

classifier = xgb.XGBClassifier(max_depth=3, n_estimators=700, learning_rate=0.05)
ensemble = classifier.fit(x, y)

#typesGrps = data.columns.to_series().groupby(data.dtypes)
#typeCounts = typesGrps.count()
#dtypes = typesGrps.groups
#{k.name: v for k, v in dtypes.items()}

#s_cat = pd.factorize(s)
# s_cat[0] - labels
# s_cat[1] - uniques

#integer = list(data.select_dtypes(include=['int64']).columns)
#floats = list(data.select_dtypes(include=['float64']).columns)
#categorical = list(data.select_dtypes(include=['object']).columns)