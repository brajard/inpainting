#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:54:29 2018
extract small images from med sea
@author: jbrajard
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

#Directory containing globcolour global images
#datadir = '/data/jbrajard/globcolor/Chl'
datadir = '/usr/home/jbrlod/data/ciclad/globcolor/Chl'
#prefix of the files to consider
prefix = 'L3m_'

#output directory to save the dataset 
#outdir ='/data/jbrajard/appledom/medchl'
outdir = '../data'

#output filename of the data
outname = 'medchl.nc'

#output filename of an ancilary file
ancname = 'medchl.csv'

#Size of images (0.04°/pixel)
ny,nx = 64,64 #3°x3°

#south,west lat,lon of images
latc = [36, 38, 40]
lonc = [1, 4, 7, 10]

#limit of valid pixel to keep image
HARD_TH = 0.95 #discard imae if proportion of invalid pixels is above

#to plot a lot of stuff
PLOT=False

# %% Extraction

#Flags definition
FINVALID = 2**0+2**1+2**3+2**4+2**5
FCLOUD = 2**4+2**5
FLAND = 2**3

#list of data files
Lfiles = glob.glob(os.path.join(datadir,prefix+'*.nc'))
#Lfiles = Lfiles[:50]
#size of the dataset
n = len(Lfiles)*len(latc)*len(lonc)

#lon,lat to extract
Latc,Lonc = np.meshgrid(latc,lonc)
#lgeo = zip(Latc.ravel(),Lonc.ravel())

#define arrays
chla = np.empty(shape=(n,ny,nx))
lon = np.empty(shape=(n,nx))
lat = np.empty(shape=(n,ny))
invalidmask = np.empty(shape=(n,ny,nx),dtype=bool)
flags = np.empty(shape=(n,ny,nx),dtype=int)
keep = np.empty(shape=(n),dtype=bool)

#define ancillary
danc = dict()
danc['index']=range(n)
danc['filename']=[]
danc['per_cloud']=np.empty(shape=(n))
danc['per_land']=np.empty(shape=(n))
danc['per_invalid']=np.empty(shape=(n))


i = 0
print('loop over files')
for fname in tqdm(Lfiles):
#for fname in Lfiles:
    data = xr.open_dataset(fname)
    #extract data
    chla1 = data['CHL-OC5_mean']
    flags1 = data['CHL-OC5_flags'].astype(int)
    lon1 = data['lon']
    lat1 = data['lat']
    
    
    #extraxt ilmages
    for (llat,llon) in zip(Latc.ravel(),Lonc.ravel()):
        #find index
        ilon = int(np.argmin((lon1-llon)**2))
        ilat = int(np.argmin((lat1-llat)**2))
        slon = slice(ilon,ilon+nx)
        slat = slice(ilat,ilat+ny)
        
        #update arrays
        chla[i,:,:]=chla1[slat,slon]
        lon[i,:]=lon1[slon]
        lat[i,:]=lat1[slat]
        flags[i,:,:]=flags1[slat,slon]
        invalidmask[i,:,:]=(flags[i,:,:] & FINVALID)>0
        
        if PLOT:
            extent = [lon1[ilon],lon1[ilon+nx],lat1[ilat],lat1[ilat+ny]]
            plt.imshow(np.log10(chla1[slat,slon]),extent =extent)
            plt.show()
            
        #update ancillary
        danc['filename'].append( fname)
        clouds = (flags[i]&FCLOUD)>0
        lands = (flags[i]&FLAND)>0
        
        danc['per_cloud'][i] =  sum(clouds.ravel())/clouds.size
        danc['per_land'][i] =  sum(lands.ravel())/lands.size
        danc['per_invalid'][i] = sum(invalidmask[i].ravel())/invalidmask[i].size
        
        if danc['per_invalid'][i] < HARD_TH:
            keep[i] = True
        else:
            keep[i] = False
        #increment index
        i+=1
chla[invalidmask] = np.nan
 #%% Createoutput structures
print('create structures')
danc = {k:np.extract(keep,danc[k]) for k in danc }
df_anc = pd.DataFrame(danc)
ds_data = xr.Dataset({'chla':(['index','y','x'], chla[keep]),
                   'flags':(['index','y','x'],flags[keep]),
                   'invalidmask':(['index','y','x'],invalidmask[keep]),
                   'lon':(['index','x'],lon[keep]),
                   'lat':(['index','y'],lat[keep])},
                    coords={'index':danc['index'],
                            'y':range(ny),
                            'x':range(nx)})
print('save file')
df_anc.to_csv(os.path.join(outdir,ancname))
ds_data.to_netcdf(os.path.join(outdir,outname))
print('Finished')

                     
