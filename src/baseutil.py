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
from random import randint
import skimage
import skimage.morphology
from copy import copy

SAVE = False
def nan_counter(ny,nx,dst,ind1, delta, lim, msize, min_MaskInvPixel):
    x = randint(lim,(nx-(msize+2*delta+lim)))
    y = randint(lim,(ny-(msize+2*delta+lim)))
    pix_max = np.array(dst.chla[ind1,y:y+msize, x:x+msize], dtype='float')
    a = np.reshape(pix_max, (1, msize*msize))
    max_MaskInvPixel = np.sum(np.isnan(a))
    return max_MaskInvPixel, x, y

def make_mask_squares(ny,nx,dst,ind1, delta=2, lim=0, msize=8, nmask = 5, min_MaskInvPixel=0,weight_c=0.5,weight_n=1 ):
    a_mask = np.zeros((ny,nx),dtype=bool)  # initialisation du masque d'extraction
    c_mask = np.ones((ny,nx),dtype=bool)   # initialisation du masque contextuel
    n_mask = np.zeros((ny,nx),dtype=bool)  # initialisation du masque de voisinage
    weights = np.empty_like(a_mask, dtype=float) # initialisation du masque de poids
    k=0
    while (k<nmask):
        max_MaskInvPixel,x ,y = nan_counter(ny,nx,dst,ind1,delta, lim, msize, min_MaskInvPixel)
        random_size = randint(0,msize)
        while (max_MaskInvPixel > min_MaskInvPixel):
            max_MaskInvPixel,x,y = nan_counter(ny,nx,dst,ind1, delta, lim, msize, min_MaskInvPixel)
        a_mask[y:(y+random_size), x:(x+random_size)] = True
        c_mask[(y-delta):(y+random_size+delta), (x-delta):(x+random_size+delta)] = False
        # Construction du masque de voisinage
        n_mask[(y-delta):(y+random_size+delta), (x-delta):(x+random_size+delta)] = True
        n_mask[y:(y+random_size), x:(x+random_size)] = False
        # Construction du masque de poids
        weights[np.where(a_mask==True)]=weight_c
        weights[np.where(n_mask==True)]=weight_n
        weights[np.where(c_mask==True)]=0
        k=k+1
    return a_mask, c_mask, n_mask, weights

def make_mask_clouds(ny,nx,dst,ind1, delta=2, lim=0, msize=8, nmask = 5, min_MaskInvPixel=0,weight_c=0.5,weight_n=1 ):
    nanval=-1e5
    dd=xr.open_dataarray("../data/data_mask.nc")
    a_mask = np.zeros((ny,nx),dtype=bool)  # initialisation du masque d'extraction
    c_mask_inv = np.zeros((ny,nx),dtype=bool)   # initialisation du masque contextuel
    n_mask = np.zeros((ny,nx),dtype=bool)  # initialisation du masque de voisinage
    weights = np.empty_like(a_mask, dtype=float) # initialisation du masque de poids
    cloud_mask=np.zeros((ny,nx),dtype=bool)
    k=0
    mn = randint(1,nmask)
    while (k<mn):
        max_MaskInvPixel = 100
        while (max_MaskInvPixel > min_MaskInvPixel):
            h = randint(0,np.shape(dd)[0]-1)
            cloud_mask=dd.rename({'dim_1': 'y','dim_2': 'x'})[h,:,:]
            # Construction du masque contextuel par dilatation du amask
            if (delta==0):
                abc = copy(cloud_mask)
            elif (delta==1):
                abc=skimage.morphology.dilation(cloud_mask, shift_x=True, shift_y=True)
            elif (delta>1):
                abc=skimage.morphology.dilation(cloud_mask, shift_x=True, shift_y=True)
                for i in range(delta-1):
                    abc=skimage.morphology.dilation(abc,shift_x=True, shift_y=True) 
            pix = dst.chla.fillna(nanval)[ind1,:,:]
            a = pix.where(abc==True)
            max_MaskInvPixel = np.sum(np.where(a<0))
        a_mask = a_mask+copy(cloud_mask)
        c_mask_inv = c_mask_inv+abc
        k=k+1
    c_mask = np.logical_not(c_mask_inv)
    # Construction du masque de voisinage
    n_mask = np.add(a_mask, c_mask)
    n_mask = ~n_mask
    # Construction du masque de poids
    weights[np.where(a_mask==True)]=weight_c
    weights[np.where(n_mask==True)]=weight_n
    weights[np.where(c_mask==True)]=0
    return a_mask, c_mask, n_mask, weights

def weights_mask(inputds,weightBaseName,weight_c,weight_n):
    #inputds = '../data/cloud/BaseTest_cloud.nc'
    #weightBaseName = '../data/cloud/weights_mask_cloud.nc'
    ds = xr.open_dataset(inputds)
    am = np.array(ds.amask.values, dtype=int)
    cm = np.array(ds.cmask.values, dtype=int)
    weights = np.empty_like(ds.amask.values, dtype=float)
    # Construction du masque de voisinage
    nm = np.logical_not(np.add(am, cm))
    # Construction du masque de poids
    if np.array_equal(am, np.logical_not(cm)):
        weights[np.where(am==1)]=weight_c
        weights[np.where(cm==1)]=0
    else:
        # Construction du masque de poids
        weights[np.where(am==1)]=weight_c
        weights[np.where(nm==1)]=weight_n
        weights[np.where(cm==1)]=0
    # Stockage dans une base de données
    AM = ds['amask']; AM.values = am 
    CM = ds['cmask']; CM.values = cm 
    NM = copy(ds['cmask']); NM.values = nm
    W = copy(ds['X']); W.values = weights
    wds = xr.Dataset({'X':(['index','y','x'],ds.X),
                      'Y':(['index','y','x'],ds.yt),
                      'weights':(['index','y','x'],W),
                      'amask':(['index','y','x'],AM),
                      'cmask':(['index','y','x'],CM),
                      'bmask':(['index','y','x'],ds.bmask),
                      'nmask':(['index','y','x'],nm)},
                      coords = ds.coords)  
    wds.to_netcdf(weightBaseName)
    return wds

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
            self._bmask = self._trainingset['bmask']
            self._cmask = self._trainingset['cmask']
            
    def masking(self, mfun=make_mask_squares, **margs):
        self._X = np.ma.masked_invalid(self._base[self._fname])
        self._yt = np.ma.masked_invalid(self._base[self._fname])
        self._amask = np.zeros(self._X.shape,dtype=bool)  # definition du training mask
        self._cmask = np.zeros(self._yt.shape,dtype=bool) # definition du contextual mask
        self._nmask = np.zeros(self._yt.shape,dtype=bool) # definition du masque du voisinage uniquement
        self._weights =np.zeros(self._yt.shape,dtype=float)
        self._bmask = ~self._X.mask
        if self._crop>0:
            self._yt = self._yt[:,self._crop:-self._crop,self._crop:-self._crop]
        for i in range(self._X.shape[0]):           
            a_m, c_m, n_m, w_m = mfun(self._ny, self._nx,self._base, i, **margs)
            self._X[i,a_m] = np.ma.masked
            self._amask[i,:,:] = a_m
            self._yt[i,c_m] = np.ma.masked
            self._cmask[i,:,:] = c_m
            self._nmask[i,:,:] = n_m
            self._weights[i,:,:] = w_m
            print("{} / {}".format(i, self._X.shape[0]-1))
            
    def savebase(self,basename=None):
        if basename is None:
            basename = self._basename
        if basename is None:
            raise ValueError('name of the file to save is not specified')
        
        self._trainingset = xr.Dataset({'X':(['index','y','x'],self._X),
                                       'yt':(['index','y','x'],self._yt),
                                       'amask':(['index','y','x'],self._amask),
                                       'bmask':(['index','y','x'],self._bmask),
                                       'cmask':(['index','y','x'],self._cmask),
                                       'weights':(['index','y','x'],self._weights),
                                       'nmask':(['index','y','x'],self._nmask)},
                                        coords = self._base.coords)  
        self._trainingset.to_netcdf(basename)
    
    @property
    def X(self):
        X = self._X.expand_dims('canal',3).fillna(0)   
        return X

    @property
    def X_2D(self):
        X = self._X.expand_dims('canal',3).fillna(0)
        X = xr.concat((X, X),dim='canal') # l'image d'entrée a 2 canaux
 
    @property
    def Xlog(self):
        Xlog = np.log10(self._X)
        Xlog = Xlog.expand_dims('canal',3).fillna(Xlog.mean())

        return Xlog
    
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
    
    @property
    def Weights(self):
        Weights = self._weights.expand_dims('canal',3)
        return Weights 
    
    @property
    def ytlog(self):
        ytlog = np.log10(self._yt)
        ytlog = ytlog.expand_dims('canal',3).fillna(self._nanval)
        
        return ytlog
    
    @property
    def Xstandard(self):
        Xstandard = self._X.expand_dims('canal',3)
        ii=0
        while (ii<self._n):
            meanImage = self._X[ii,:,:].mean()
            stdImage = self._X[ii,:,:].std()
            Xstandard[ii,:,:,0] = (self._X[ii,:,:]-meanImage)/stdImage
            ii += 1
      
        return Xstandard
    
    @property
    def bmask(self):
        bmask = self._bmask.expand_dims('canal',3)   
        return bmask
    
if __name__  == "__main__":
    import matplotlib.pyplot as plt
    fname = '../data/base_mini.nc'
    fout = '../data/trainingset-small.nc'
    outdir = '../figures/examples'
    ds = dataset(srcname = fname, overwrite = True)
    ds.masking(mfun=make_mask_squares)
    ds.savebase(fout)
    
    nim= 20
    
    
    ii = np.random.randint(0,ds._n,nim)
    
    for i,ind in enumerate(ii):
        fig, axes= plt.subplots(ncols=5)
        axes[0].imshow(np.log10(ds._X[ind,:,:]))
        axes[1].imshow(np.log10(ds._yt[ind,:,:]))
        axes[2].imshow(ds._bmask[ind,:,:],cmap=plt.get_cmap('binary'))
        axes[3].imshow(ds._nmask[ind,:,:],cmap=plt.get_cmap('binary'))
        axes[4].imshow(ds._weights[ind,:,:])
        title = 'Image_' + str(int(ds._base.index[ind]))
        plt.suptitle(title)
        if SAVE:
            plt.savefig(os.path.join(outdir,title+'.png'))
        
