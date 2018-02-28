#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to create the training set
@author: Julien Brajard
"""
from baseutil import dataset, make_mask
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


#data directory
datadir = '../data'

# name of the input data
basename = 'medchl-small.nc'

# name of the output training dataset
trainingname = 'training-small.nc'

# option for masking
mfun = make_mask #masking function (in baseutil)
margs = \
   {'msize':8, #size of the mask
    'nmask':1} #number of mask per image



#Make the dateset
ds = dataset(srcname = os.path.join(datadir,basename) , overwrite = True)
ds.masking(mfun = mfun, **margs)
ds.savebase(os.path.join(datadir,trainingname))

#plot some random images
PLOT = True

#save the images
SAVE = True

#Plot some images
if PLOT:
    # example dir
    exampledir = os.path.join('../figures/examples/',os.path.splitext(trainingname)[0])
    shutil.rmtree(exampledir,ignore_errors=True) 

    os.makedirs(exampledir)
    
    nim = 20 #number of images to be plot
    ii = np.random.randint(0,ds._n,nim)
    
    for i,ind in enumerate(ii):
        fig, axes= plt.subplots(ncols=3)
        axes[0].imshow(np.log10(ds._X[ind,:,:]))
        axes[1].imshow(np.log10(ds._yt[ind,:,:]))
        axes[2].imshow(ds._amask[ind,:,:],cmap=plt.get_cmap('binary'))
        title = 'Image_' + str(int(ds._base.index[ind]))
        plt.suptitle(title)
        if SAVE:
            plt.savefig(os.path.join(exampledir,title+'.png'))

    
