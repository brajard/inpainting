#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:16:07 2018

@author: Julien Brajard
"""

# %% Import
import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
# %% Parameters
outdir = '../data'

#output filename of the data
ancname = 'liban.nc'
reconstruct_name = '/net/argos/data/parvati/arimoux/share/liban/liban_completed.nc'
meshinfo = 'meshinfo_liban.npz'
outfile = 'liban_rec.nc'

#Size of images (0.04°/pixel)
ny,nx = 64,64 #3°x3°

#overlap
dy,dx = 8,8

#to plot a lot of stuff
PLOT=True

danc = xr.open_dataset(os.path.join(outdir,ancname))
dr = xr.open_dataset(reconstruct_name)
dmesh = np.load(os.path.join(outdir,meshinfo))


#list of days
lofdays = np.unique(np.sort([int(ind//100) for ind in danc.index]))

#mask lon/lat
mlon = np.logical_and(dmesh['lon']>dmesh['lonc'][0],
                     dmesh['lon']<dmesh['lonc'][1])

mlat = np.logical_and(dmesh['lat']>dmesh['latc'][0],
                     dmesh['lat']<dmesh['latc'][1])

lon_e = dmesh['lon'][mlon]
lat_e = dmesh['lat'][mlat]

iLatc = dmesh['iLatc']
iLonc = dmesh['iLonc']

# %% Prepare extraction
chla = dict()
chla['rec'] = np.empty(shape=(len(lofdays),len(lat_e),len(lon_e)))
chla['raw'] = np.empty(shape=(len(lofdays),len(lat_e),len(lon_e)))

# %% Extractop,
iday = 0
for day in tqdm(lofdays):
    mask = np.array([(d//100) == day for d in danc.index])
    y = dict()
    y['rec'] = dr.yfinal[mask][:,dx//2:-dx//2,dy//2:-dy//2]
    y['raw'] = danc.chla[mask][:,dx//2:-dx//2,dy//2:-dy//2]
    tmp = dict()
    for k in y:
        tmp[k] = np.zeros(shape=(len(dmesh['lat']),len(dmesh['lon'])))
    i=0
    for (ilat,ilon) in zip(iLatc.ravel(),iLonc.ravel()):
        slon = slice(ilon+dy//2,ilon+nx-dy//2)
        slat = slice(ilat+dx//2,ilat+ny-dx//2)
        for k in y:
            tmp[k][slat,slon]=y[k][i].squeeze()
        i+=1
    
    
    for k in y:
        chla[k][iday] = tmp[k][mlat,:][:,mlon]
        
    plt.imshow(np.log10(chla['rec'][iday]))
    plt.show()
    iday += 1
    
    
#%% Save data
dout = xr.Dataset({'chla_rec':(['index','lat','lon'], chla['rec']),
                   'chla_raw':(['index','lat','lon'], chla['raw'])},
                    coords = {'index':lofdays,
                              'lat':lat_e,
                              'lon':lon_e})
dout.to_netcdf(os.path.join(outdir,outfile))
