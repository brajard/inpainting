#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:16:07 2018

@author: arimoux and J. Brajard
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
datadir = '/net/argos/data/parvati/jbrlod/data/share/globcolor/Chl'
#prefix of the files to consider
prefix = 'L3m_2013'
months = ['08','09']

#output directory to save the dataset 
#outdir ='/data/jbrajard/appledom/medchl'
outdir = '../data'

#output filename of the data
outname = 'liban.nc'

#meshinfo
meshinfo = 'meshinfo_liban.npz'

#output filename of an ancilary file
ancname = 'liban.csv'

#Size of images (0.04°/pixel)
ny,nx = 64,64 #3°x3°

#overlap
dy,dx = 8,8

#south,west lat,lon of images
latc = [31.7, 35.7]
lonc = [30.5, 36]

#to plot a lot of stuff
PLOT=True

# %% Prepare extraction

#Flags definition
FINVALID = 2**0+2**1+2**3+2**4+2**5
FCLOUD = 2**4+2**5
FLAND = 2**3

#list of data files
Lfiles = glob.glob(os.path.join(datadir,prefix+months[0]+'*.nc'))
Lfiles2 = glob.glob(os.path.join(datadir,prefix+months[1]+'*.nc'))
Lfiles.extend(Lfiles2)
Lfiles.sort()

indexmin=24 #start date - 25/08
indexmax=-1 #end date - 29/09
Lfiles = Lfiles[indexmin:indexmax]

#lon,lat to extract
Latc,Lonc = np.meshgrid(latc,lonc)
#lgeo = zip(Latc.ravel(),Lonc.ravel())

data = xr.open_dataset(Lfiles[0])

lon1 = data['lon']
lat1 = data['lat']
ilatmax = int(np.argmin((lat1-latc[0])**2))
ilatmin = int(np.argmin((lat1-latc[1])**2))
ilonmin = int(np.argmin((lon1-lonc[0])**2))
ilonmax = int(np.argmin((lon1-lonc[1])**2))


ilat_list = np.arange(ilatmin-dy,ilatmax+dy,ny-dy)
ilon_list = np.arange(ilonmin-dx,ilonmax+dx,nx-dx)

iLatc, iLonc = np.meshgrid(ilat_list,ilon_list)
np.savez(os.path.join(outdir,meshinfo),
         lon=lon1,
         lat=lat1,
         iLonc=iLonc,
         iLatc=iLatc,
         latc=latc,
         lonc=lonc)

# %% Ectract dataset
#size of the dataset
n = len(Lfiles)*len(ilat_list)*len(ilon_list)

ifiles = range(len(Lfiles))
igeo = range(iLatc.size)

indg,indf = np.meshgrid(igeo,ifiles)
ind = 100*indf.ravel() + indg.ravel()
#define arrays
chla = np.empty(shape=(n,ny,nx))
lon = np.empty(shape=(n,nx))
lat = np.empty(shape=(n,ny))
invalidmask = np.empty(shape=(n,ny,nx),dtype=bool)
flags = np.empty(shape=(n,ny,nx),dtype=int)
keep = np.empty(shape=(n),dtype=bool)

#define ancillary
danc = dict()
danc['index']=ind
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
        
    for (ilat,ilon) in zip(iLatc.ravel(),iLonc.ravel()):
        slon = slice(ilon,ilon+nx)
        slat = slice(ilat,ilat+ny)
        
        #update arrays
        chla[i,:,:]=chla1[slat,slon]
        lon[i,:]=lon1[slon]
        lat[i,:]=lat1[slat]
        flags[i,:,:]=flags1[slat,slon]
        invalidmask[i,:,:]=(flags[i,:,:] & FINVALID)>0
        danc['index'][i] = 100*int(data.period_start_day)+danc['index'][i]%100
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
        
        keep[i] = True
        
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

#plot to check
if PLOT:
    plt.imshow(np.log10(chla1[ilat_list[0]:ilat_list[-1]+64,ilon_list[0]:ilon_list[-1]+64]))
                    
