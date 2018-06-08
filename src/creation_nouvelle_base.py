# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:56:55 2018

pour faire une nouvelle base en jouant sur le pourcentage des valeurs manquantes 

err_cloud
err_invalide
err_land

@author: DELL
"""
import numpy as np
import xarray as xr
from numpy import genfromtxt
from sklearn.preprocessing import Imputer

imp = Imputer(strategy="mean")
ds = xr.open_dataset('../data/medchl.nc')
data = genfromtxt('../data/medchl.csv', delimiter=',')

ny,nx = 64,64 #3°x3°

duplicate=True
repetition=3 # number of database repetition 
err_min=0.1
err_max=0.3
err_cloud=0.2
err_invalide=0.2
err_land=0.01


a=[]
for i in range(len(data[:,1])-1):
    if ((data[i+1,3]<=err_max) & (data[i+1,4]<=err_max) & (data[i+1,5]<=err_max) & (data[i+1,3]>=err_min) & (data[i+1,4]>=err_min) & (data[i+1,5]>=err_min)):
        a.append(i)

keep = np.zeros(shape=ds.dims['index']).astype(bool)
keep[[a]] = True

chlanew = ds.chla[keep]
flagnew= ds.flags[keep]
ds_new = xr.Dataset({'chla':chlanew,'flags':flagnew})

if duplicate:
    ds_k=ds_new
    for k in range(1,repetition,1):
        ds_k.index.values= ds_k.index.values +k*1000000
        ds_new=xr.concat([ds_new, ds_k], 'index')
        
ds_new.to_netcdf('../data/base_'+str(int(err_min*100))+'_'+str(int(err_max*100))+'.nc')

import matplotlib.pyplot as plt
ind=2
fig, axes= plt.subplots(ncols=1)
axes.imshow(np.log10(ds_new.chla[ind,:,:]))
