#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:06:35 2018
classes and utils for dealing with training dataset
@author: jbrlod
"""
import os
import xarray as xr
import numpy as np


SAVE = False

def make_mask (ny,nx, msize=8, nmask = 1):
    assert nmask==1,'nmask different of 1 not implemented yet'
    x = np.random.randint(0,nx-msize)
    y = np.random.randint(0,ny-msize)
    mask = np.zeros((ny,nx),dtype=bool)
    mask[y:y+msize,x:x+msize] = True
    return (mask)



class dataset:
    def __init__(self, srcname = None, basename = None, crop = 0,
                 overwrite=False,fname='chla',nanval=-1e5):
        self._overwrite = overwrite
        self._crop = crop
        self._fname = fname
        self._basename = basename
        self._nanval = nanval
        if not srcname is None :
            if not basename is None :
                if os.path.exists(basename) and not overwrite:
                    raise ValueError(basename + ' exists and can not be overwritten')
            self._base = xr.open_dataset(srcname)
            self._nx = self._base.dims['x']
            self._ny = self._base.dims['y']
            self._n = self._base.dims['index']
         
        else :
            self._base=None
            self._trainingset = xr.open_dataset(basename)
            self._X = self._trainingset['X']
            self._yt = self._trainingset['yt']
            self._amask = self._trainingset['amask']
            self._nx = self._trainingset.dims['x']
            self._ny = self._trainingset.dims['y']
            self._n = self._trainingset.dims['index']
        
    def masking(self, mfun=make_mask,**margs):
        self._X = np.ma.masked_invalid(self._base[self._fname])
        self._yt = np.ma.masked_invalid(self._base[self._fname])
        self._amask = np.zeros(self._X.shape,dtype=bool)
        if self._crop>0:
            self._yt = self._yt[:,self._crop:-self._crop,self._crop:-self._crop]
        for i in range(self._X.shape[0]):
            m = mfun(self._ny,self._nx,**margs)
            self._X[i,m] = np.ma.masked
            self._amask[i,:,:] = m 
            
    def savebase(self,basename=None):
        if basename is None:
            basename = self._basename
        if basename is None:
            raise ValueError('name of the file to save is not specified')
        
        self._trainingset = xr.Dataset({'X':(['index','y','x'],self._X),
                                       'yt':(['index','y','x'],self._yt),
                                       'amask':(['index','y','x'],self._amask)},
                                        coords = self._base.coords)  
        self._trainingset.to_netcdf(basename)
    
    @property
    def X(self):
        X = self._X.expand_dims('canal',3).fillna(0)
        
        return X
 
    @property
    def Xmasked(self):
        return np.ma.masked_invalid(self._X)
    
    def ymasked(self,y=None):
        if y is None:
            y = self._yt
        return np.ma.masked_invalid(y)
    
    @property
    def yt(self):
        yt = self._yt.expand_dims('canal',3).fillna(self._nanval)
        
        return yt

if __name__  == "__main__":
    import matplotlib.pyplot as plt
    fname = '../data/medchl-small.nc'
    fout = '../data/trainingset-small.nc'
    outdir = '../figures/examples'
    ds = dataset(srcname = fname, overwrite = True)
    ds.masking()
    ds.savebase(fout)
    nim= 20
    
    
    ii = np.random.randint(0,ds._n,nim)
    
    for i,ind in enumerate(ii):
        fig, axes= plt.subplots(ncols=3)
        axes[0].imshow(np.log10(ds._X[ind,:,:]))
        axes[1].imshow(np.log10(ds._yt[ind,:,:]))
        axes[2].imshow(ds._amask[ind,:,:],cmap=plt.get_cmap('binary'))
        title = 'Image_' + str(int(ds._base.index[ind]))
        plt.suptitle(title)
        if SAVE:
            plt.savefig(os.path.join(outdir,title+'.png'))
