# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:45:49 2015

@author: bolaka
"""

import pandas as pd
import graphlab as gl
from sklearn.metrics import roc_curve, auc
import gc
import os
from glob import glob

#chunksize = 7000
#dtypes = []
#dirs = ['~/datasets/Springleaf/train.csv', '~/datasets/Springleaf/test.csv']
#for ifile in dirs:
#    counter = 0
#    prefix = ifile.replace('.csv', '').split('/')[3]
#    print 'chunking ' + prefix + '...'
#    
#    for chunk in pd.read_csv(ifile, chunksize=chunksize):
#        chunk.to_csv(prefix + '_chunks/' + prefix + '_' + str(counter) + '.csv', index=False)
#        counter += 1
#
#    files_list = glob(os.path.join(prefix + '_chunks/', '*.csv'))
#    file_orders = [(int(file_i.replace('.csv', '').split('_')[2]), file_i) for file_i in files_list]
#    file_orders.sort(key=lambda x: x[0])
#    
#    for i, file_name in file_orders:
#        print i, file_name
#        if i == 0:
#            if prefix == 'train':
#                chunk = gl.SFrame.read_csv(file_name, header=True, verbose=True)
#                dtypes.extend(chunk.column_types())
#            else:
#                dtypes.pop()
#                chunk = gl.SFrame.read_csv(file_name, header=True, verbose=True, column_type_hints=dtypes)
#                
#            whole = gl.SFrame(chunk)
#        else:
#            chunk = gl.SFrame.read_csv(file_name, header=True, verbose=True, column_type_hints=dtypes)
#            whole = whole.append(chunk)
#        print whole.shape
#    
#    whole.save('~/datasets/Springleaf/' + prefix +'_binary')
    
#        unmatches = [i for i, j in zip(dtypes, tocheck) if i != j]
#'  '.join(data.column_names())

data = gl.load_sframe('~/datasets/Springleaf/train_binary')
test = gl.load_sframe('~/datasets/Springleaf/test_binary')

# row & column count
print 'training data has ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns'
print 'test data has ' + str(test.shape[0]) + ' rows and ' + str(test.shape[1]) + ' columns'

# Pre-process data
## add the target columns to test data as 9999
#test['target'] = 9999
#
## combine the training and test datasets for data preprocessing
#combined = data.append(test)

counter = 0
for i in data.column_names():
    uniques = data[i].unique()    
    dtype = data[i].dtype()
    if (dtype is int) or (dtype is float):
        variance = data[i].var()
        std = data[i].std()
        if variance == 0:
            print i
            data.remove_column(i)
            test.remove_column(i)
            counter += 1
    elif dtype is str: #  and uniques.size() == 1
        print i
        data.remove_column(i)
        test.remove_column(i)
        counter += 1
    elif uniques.size() == 2 and uniques.any() == False:
        print i
        data.remove_column(i)
        test.remove_column(i)
        counter += 1
    gc.collect()    

print 'removed ' + str(counter) + ' columns.'
print 'now data has ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns'

## separate again into training and test sets
#data = combined[ combined['target'] != 9999 ]
#test = combined[ combined['target'] == 9999 ]
#
## remove the target columns from test data
#test.remove_column('target')

#def modeling(data, test):
#data = data.sample(0.1, 786)

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Create a model.
model = gl.boosted_trees_classifier.create(train_data, target='target', class_weights = 'auto')

# Save predictions to an SFrame (class and corresponding class-probabilities)
predictions = model.classify(test_data)

fpr, tpr, _ = roc_curve(test_data['target'], predictions['probability'])
roc_auc = auc(fpr, tpr)
print('--- test ROC_AUC = ', roc_auc) 
# --- 0.44208 on LB, 0.43718 on public LB ---
# --- 0.49099 on LB, 0.48988 on public LB ---

test_predictions = model.classify(test)
    
# test_predictions['probability']

test['target'] = test_predictions['probability']
test[['ID', 'target']].save('~/datasets/Springleaf/graphlab_submission', format='csv')
   


## Evaluate the model and save the results into a dictionary
#results = model.evaluate(test_data)