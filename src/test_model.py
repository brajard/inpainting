#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:31:46 2018

@author: jbrlod
"""

from baseutil import dataset
ds = dataset(basename='../data/trainingset-small.nc')
from keras.models import load_model
from modelutil import masked_mse
import os
import numpy as np
import matplotlib.pyplot as plt

SAVE= True

name = 'model_7layers'

model = load_model(name,custom_objects={'masked_mse':masked_mse})

ypredict = model.predict(ds.X).squeeze()

nim = 20
ii = np.random.randint(0,ds._n,nim)
vmin, vmax = -1.5,1
outdir = '../figures/examples_app'
for i,ind in enumerate(ii):
    fig, axes= plt.subplots(ncols=3)
    axes[0].imshow(np.log10(ds.Xmasked[ind]),vmin=vmin,vmax=vmax)
    axes[1].imshow(np.log10(ds.ymasked()[ind]),vmin=vmin,vmax=vmax)
    axes[2].imshow(np.log10(ds.ymasked(ypredict)[ind]),vmin=vmin,vmax=vmax)
    title = 'Image_' + str(int(ds._trainingset.index[ind]))
    plt.suptitle(title)
    if SAVE:
        plt.savefig(os.path.join(outdir,title+'.png'))