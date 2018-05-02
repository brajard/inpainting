#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 13:59:36 2018

@author: arimoux
"""

import os
from baseutil import dataset

#change  the trainingset name  if necessary
trainingname = 'trainingset-small.nc'

#data directory
datadir = '../data/'

ds = dataset(basename=os.path.join(datadir, trainingname))

# model tuning
from sklearn.metrics import make_scorer
from keras.wrappers.scikit_learn import KerasRegressor
from modelutil import get_model_4layers

model = KerasRegressor(build_fn=get_model_4layers,  epochs=30, batch_size=10, verbose=False)

# define the grid search parameters
ind_param = 0
# Set the parameters by cross-validation
if ind_param==0:
    batch_size = [10, 20, 40, 60, 80, 100]
    epochs = [10, 30, 50]
    tuned_parameters = dict(batch_size=batch_size, epochs=epochs)
elif ind_param==1:
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    tuned_parameters = dict(optimizer=optimizer)
elif ind_param==2:
    learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    tuned_parameters = dict(learn_rate=learn_rate,momentum=momentum)
elif ind_param==3:
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    tuned_parameters = dict(init_mode=init_mode)
elif ind_param==4:
    filter_number= [16, 32, 64]
    tuned_parameters = dict(filter_number=filter_number)
elif ind_param==5:
    kernel_size=[(2,2), (3,3), (4,4), (5,5), (6,6), (7,7)]
    tuned_parameters = dict(kernel_size=kernel_size)
elif ind_param==6:
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    tuned_parameters = dict(activation=activation)
    
#%%
from sklearn.model_selection import train_test_split,GridSearchCV
#loss = make_scorer(model.score,greater_is_better=False)
grid = GridSearchCV(estimator=model, param_grid=tuned_parameters, n_jobs=1, verbose=2)
X_train, X_test, yt_train, yt_test = train_test_split(ds.X, ds.yt, test_size=0.2)
grid_result = grid.fit(X_train, yt_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

#%%
filename = "../data/gridSearch_"+str(ind_param)+".txt"
file = open(filename,"w") 
for mean, stdev, param in zip(means, stds, params):
    print("mean: %f (std: %f) with: %r" % (mean, stdev, param))
    file.write("mean: %f (std: %f) with: %r \n" % (mean, stdev, param))

print("best score: %f (std: %f) for paramaters: %r" % (means[means.argmin()], stds[means.argmin()], params[means.argmin()]))  
file.write("best score: %f (std: %f) for paramaters: %r \n" % (means[means.argmin()], stds[means.argmin()], params[means.argmin()]))
file.close() 
