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
import skimage
import skimage.morphology
from copy import copy

SAVE = False
def nan_counter(ny,nx,dst,ind1,dy, dx, y_lim, x_lim, msize, min_MaskInvPixel):
    x = np.random.randint(x_lim,(nx-(msize+x_lim)))
    y = np.random.randint(y_lim,(ny-(msize+y_lim)))
    pix_max = np.array(dst.chla[ind1,y:y+msize, x:x+msize], dtype='float')
    a = np.reshape(pix_max, (1, msize*msize))
    max_MaskInvPixel = np.sum(np.isnan(a))
    return max_MaskInvPixel, x,y

def make_mask(ny,nx,dst,ind1, dy=0,dx=0, y_lim=0, x_lim=0, msize=8, nmask = 1, min_MaskInvPixel=3 ):
    assert nmask==1,'nmask different of 1 not implemented yet'
    print("Etape 1")
    max_MaskInvPixel,x,y = nan_counter(ny,nx,dst,ind1,dy, dx, y_lim, x_lim, msize, min_MaskInvPixel)
    count =0
    while ((max_MaskInvPixel > min_MaskInvPixel) and (count<20)):
        max_MaskInvPixel,x,y = nan_counter(ny,nx,dst,ind1,dy, dx, y_lim, x_lim, msize, min_MaskInvPixel)
        count = count+1
        print(count)
    # construction du masque d'extraction de la region à reconstruire    
    a_mask = np.zeros((ny,nx),dtype=bool)  # initialisation du trainig mask
    a_mask[y:(y+msize), x:(x+msize)] = 1
    # Construction du contextual mask
    c_mask = np.ones((ny,nx),dtype=bool)   # initialisation du contextual mask
    c_mask[(y-dy):(y+msize+dy), (x-dx):(x+msize+dx)] = 0
    # Construction du masque du voisinage 
    n_mask = np.zeros((ny,nx),dtype=bool)
    n_mask[(y-dy):(y+msize+dy), (x-dx):(x+msize+dx)] = 1
    n_mask[y:(y+msize), x:(x+msize)] = 0
    # Construction du masque de poids
    weights = np.empty_like(a_mask, dtype=float)
    weights[np.where(a_mask==1)]=0.5
    weights[np.where(n_mask==1)]=1
    weights[np.where(c_mask==1)]=0
    return a_mask, c_mask ,n_mask, weights

def make_mask3(ny, nx, dst, ind1, dx=0 , nmask = 1, min_MaskInvPixel=10):
    """ dx est le nombre de dilatation lors de la création du contextual mask """
    assert nmask==1,'nmask different of 1 not implemented yet'
    cl_mask=np.zeros((ny,nx),dtype=int) # cloud's mask initialization
    md=xr.open_dataarray("../data/data/data_mask.nc")
    idx = np.random.randint(1,np.shape(md)[0])
    cl_mask=md[idx,:,:]
    count =0
    max_MaskInvPixel=4000
    print("Etape 1")
    while ((max_MaskInvPixel > min_MaskInvPixel) and (count<20)):
        count = count+1;
        idx= np.random.randint(1,np.shape(md)[0])
        cl_mask=np.zeros((ny,nx),dtype=int)
        cl_mask=md[idx,:,:]
        pix = np.array(dst.chla[ind1,:,:], dtype='float')
        a = np.where(np.isnan(pix))
        print("Etape 2") 
        max_MaskInvPixel = np.count_nonzero((cl_mask[a[0],a[1]]==True))
    a_mask = np.zeros((ny,nx),dtype=int)  # initialisation du training mask
    a_mask = copy(cl_mask)
    # Construction du masque contextuel par dilatation du amask
    if (dx==0):
        abc = copy(cl_mask)
    elif (dx==1):
        abc=skimage.morphology.dilation(cl_mask, shift_x=True, shift_y=True)
    elif (dx>1):
        abc=skimage.morphology.dilation(cl_mask, shift_x=True, shift_y=True)
        for i in range(dx-1):
            abc=skimage.morphology.dilation(abc,shift_x=True, shift_y=True) 
    # Construction du "contextual mask"
    c_mask = np.ones((ny,nx),dtype=int)   # initialisation du contextual mask
    c_mask = np.logical_not(abc)
    # Construction du "neigbor mask"
    n_mask = np.add(a_mask, c_mask)       # pas besoin d'initialisation , n_mask hérite son dtype de ses arguments
    n_mask[np.where(n_mask==1)] = 2; n_mask[np.where(n_mask==0)] = 1; n_mask[np.where(n_mask==2)] = 0
    # Construction du masque de poids
    weights = np.empty_like(a_mask, dtype=float)
    weights[np.where(a_mask==1)]=0.5
    weights[np.where(n_mask==1)]=1
    weights[np.where(c_mask==1)]=0
    return a_mask, c_mask, n_mask, weights

def weights_mask(inputds,weightBaseName, coefC=0.1, coefN=1):
    #inputds = '../data/cloud/BaseTest_cloud.nc'
    #weightBaseName = '../data/cloud/weights_mask_cloud.nc'
    ds = xr.open_dataset(inputds)
    am = np.array(ds.amask.values, dtype=int) ; 
    cm = np.array(ds.cmask.values, dtype=int)
    nm = np.add(am,cm) 
    weights = np.empty_like(ds.amask.values, dtype=float)
    if np.array_equal(nm, np.ones_like(am)):
        # Construction du masque de poids
        weights[np.where(am==1)]=1
        weights[np.where(cm==1)]=0
    else:
        nm[np.where(nm==1)] = 2; nm[np.where(nm==0)] = 1; nm[np.where(nm==2)] = 0
        # Construction du masque de poids
        weights[np.where(am==1)]=coefC
        weights[np.where(nm==1)]=coefN
        weights[np.where(cm==1)]=0
    # Stockage dans une base de données
    AM = ds['amask']; AM.values = am 
    NM = ds['nmask']; NM.values = nm
    W = copy( ds['X']); W.values = weights
    wds = xr.Dataset({'X':(['index','y','x'],ds.X),
                                       'weights':(['index','y','x'],W),
                                       'amask':(['index','y','x'],AM),
                                       'nmask':(['index','y','x'],NM)},
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
            self._weights = self._trainingset['weights']
            self._nx = self._trainingset.dims['x']
            self._ny = self._trainingset.dims['y']
            self._n = self._trainingset.dims['index']
        
    def masking(self, mfun=make_mask,**margs):
        self._X = np.ma.masked_invalid(self._base[self._fname])
        self._yt = np.ma.masked_invalid(self._base[self._fname])
        self._amask = np.zeros(self._X.shape,dtype=int)
        self._cmask = np.zeros(self._yt.shape,dtype=int)
        self._weights =np.zeros(self._yt.shape,dtype=float)
        if self._crop>0:
            self._yt = self._yt[:,self._crop:-self._crop,self._crop:-self._crop]
        for i in range(self._X.shape[0]):
            a_m, c_m, n_m, w_m = mfun(self._ny, self._nx,self._base, i, **margs)
            self._X[i,a_m] = np.ma.masked
            self._amask[i,:,:] = a_m
            self._weights[i,:,:] = w_m
            print("{} / {}".format(i, self._X.shape[0]))
            
    def masking2(self, mfun=make_mask, **margs):
        self._X = np.ma.masked_invalid(self._base[self._fname])
        self._yt = np.ma.masked_invalid(self._base[self._fname])
        self._amask = np.zeros(self._X.shape,dtype=bool)  # definition du training mask
        self._cmask = np.zeros(self._yt.shape,dtype=bool) # definition du contextual mask
        self._nmask = np.zeros(self._yt.shape,dtype=bool) # definition du masque du voisinage uniquement
        self._weights =np.zeros(self._yt.shape,dtype=float)
        #self._xVect = []; self._yVect = []
        if self._crop>0:
            self._yt = self._yt[:,self._crop:-self._crop,self._crop:-self._crop]
        for i in range(self._X.shape[0]):           
            a_m, c_m, n_m, w_m = mfun(self._ny, self._nx, self._base, i, **margs)
            self._X[i, a_m] = np.ma.masked
            self._amask[i,:,:] = a_m
            self._yt[i, c_m] = np.ma.masked
            self._cmask[i,:,:] = c_m 
            self._nmask[i,:,:] = n_m
            self._weights[i,:,:] = w_m
            print("{} / {}".format(i, self._X.shape[0]))

    
    def masking3(self, mfun=make_mask3, **margs):
        self._X = np.ma.masked_invalid(self._base[self._fname])
        self._yt = np.ma.masked_invalid(self._base[self._fname])
        self._amask = np.zeros(self._X.shape,dtype=bool)  # definition du training mask
        self._cmask = np.zeros(self._yt.shape,dtype=bool) # definition du contextual mask
        self._nmask = np.zeros(self._yt.shape,dtype=bool) # definition du masque du voisinage uniquement
        self._weights =np.zeros(self._yt.shape,dtype=float)
        if self._crop>0:
            self._yt = self._yt[:,self._crop:-self._crop,self._crop:-self._crop]
        for i in range(self._X.shape[0]):           
            a_m, c_m, n_m, w_m = mfun(self._ny, self._nx, self._base, i, **margs)
            self._X[i,a_m] = np.ma.masked
            self._amask[i,:,:] = a_m
            self._yt[i,c_m] = np.ma.masked
            self._cmask[i,:,:] = c_m 
            self._nmask[i,:,:] = n_m
            self._weights[i,:,:] = w_m
            print("{} / {}".format(i, self._X.shape[0]))
			
	
    def savebase(self,basename=None):
        if basename is None:
            basename = self._basename
        if basename is None:
            raise ValueError('name of the file to save is not specified')
        
        self._trainingset = xr.Dataset({'X':(['index','y','x'],self._X),
                                       'yt':(['index','y','x'],self._yt),
                                       'amask':(['index','y','x'],self._amask),
                                       'cmask':(['index','y','x'],self._cmask),
                                       'nmask':(['index','y','x'],self._nmask)},
                                        coords = self._base.coords)  
        self._trainingset.to_netcdf(basename)
     
    @property
    def X_2D(self):
        X = self._X.expand_dims('canal',3).fillna(0)
        X = xr.concat((X, X),dim='canal') # l'image d'entrée a 2 canaux  
        return X
    
    @property
    def X(self):
        X = self._X.expand_dims('canal',3).fillna(0)
        return X
 
    @property
    def Xlog(self):
        Xlog = np.log10(self._X)
        Xlog = Xlog.expand_dims('canal',3).fillna(Xlog.mean())
        return Xlog
    
    @property
    def Xmasked(self):
        return np.ma.masked_invalid(self._X)
    
    @property
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
    def Xmean(self):
        Xmean = self._X.expand_dims('canal',3)
        meanImage = np.zeros(self._n)
        ii=0
        while (ii<self._n):
            meanImage[ii] = self._X[ii,:,:].mean()
            Xmean[ii,:,:] = Xmean[ii,:,:].fillna(meanImage[ii])
            ii += 1
      
        return Xmean

if __name__  == "__main__":
    import matplotlib.pyplot as plt
    fname = '../data/data/medchl-small.nc'
    fout = '../data/data/trainingset-small.nc'
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
        