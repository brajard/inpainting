#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:31:46 2018
script python to plot some outpus of the model
@author: jbrlod
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import xarray as xr
from baseutil import dataset

#change  the trainingset name  if necessary (can be a testing dataset)
trainingname = 'training-small.nc'

#data directory
datadir = '../data'
ds = dataset(basename=os.path.join(datadir, trainingname))

from keras.models import load_model
from modelutil import masked_mse

# name of the neural networks
name = 'model_4layers'

# outputname
outname = 'dataset_nn.nc'

model = load_model(os.path.join(datadir,name),
                   custom_objects={'masked_mse':masked_mse})

ypredict = xr.DataArray(model.predict(ds.X),coords=ds.yt.coords)

#generate a combinated image of original & predicted images
nanval = -1e-5
isCloud = np.equal(ds.X,0)
yfinal = (1-isCloud)*ds.X + isCloud*ypredict
yfinal= yfinal*~ds._landmask+nanval*ds._landmask

#save prediction
dsout = xr.Dataset({'X':(['index','y','x'],ds._X),
                    'ytfull':(['index','y','x'],ds._ytfull),
                    'ypredict':(['index','y','x'],ypredict[:,:,:,0]),
                    'yfinal':(['index','y','x'],yfinal[:,:,:,0]),
                    'amask':(['index','y','x'],ds._amask),
                    'landmask':(['index','y','x'],ds._landmask)},
                    coords = ds.yt.coords)

dsout.to_netcdf(os.path.join(datadir,outname))

#plot some random images
PLOT = True

#save the images
SAVE = True

if PLOT:
    # example dir
    exampledir = os.path.join('../figures/examples/',os.path.splitext(name)[0])
    shutil.rmtree(exampledir,ignore_errors=True) 

    os.makedirs(exampledir)
    
    nim = 20 #number of images to be plot
    ii = np.random.randint(0,dsout.index.size,nim)
    vmin, vmax = -1.5,1

    for i,ind in enumerate(ii):
        fig, axes= plt.subplots(ncols=3)
        axes[0].imshow(np.log10(dsout.X[ind].squeeze()),vmin=vmin,vmax=vmax)
        axes[1].imshow(np.log10(dsout.yt[ind].squeeze()),vmin=vmin,vmax=vmax)
        axes[2].imshow(np.log10(dsout.ypredict[ind].squeeze()),vmin=vmin,vmax=vmax)
        title = 'Image_' + str(int(dsout.index[ind]))
        plt.suptitle(title)
        if SAVE:
            plt.savefig(os.path.join(exampledir,title+'.png'))

