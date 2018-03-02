#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script python to make a training
"""
import os
from baseutil import dataset

#change  the trainingset name  if necessary
trainingname = 'training-small.nc'

#data directory
datadir = '../data'

ds = dataset(basename=os.path.join(datadir, trainingname))
#for unknown reasons, it seems to work better if dataset is instaciated
# before importing keras
from modelutil import get_model_2layers


# name of the neural networks
name = 'model_2layers'

#dimension of input data
img_rows, img_cols = ds.X.shape[1:3]

model = get_model_2layers(img_rows, img_cols)
model.fit(ds.X, ds.yt, epochs=10)

model.save(os.path.join(datadir, name))
