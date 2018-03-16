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


err_cloud=0.05
err_invalide=0.05
err_land=0.05

a=[]
for i in range(len(data[:,1])-1):
    if ((data[i+1,3]<=err_cloud)  & (data[i+1,4]<=err_invalide)& (data[i+1,5]<=err_land)):
        a.append(i)

keep = np.zeros(shape=ds.dims['index']).astype(bool)
keep[[a]] = True

chlanew = ds.chla[keep]
flagnew= ds.flags[keep]


ds_new = xr.Dataset({'chla':chlanew,'flags':flagnew})

""" apres le netoyage on duplique la base 3,4 ou 5 fois """

ds1=ds2=ds3=ds4=ds_new

ds1.index.values= ds1.index.values +1000000
ds2.index.values= ds2.index.values +2000000
ds3.index.values= ds3.index.values +3000000
ds4.index.values= ds4.index.values +4000000

ds_final=xr.concat([ds_new, ds1,ds2,ds3,ds4], 'index')
ds_final.to_netcdf('../data/nouvelle_base.nc')
