#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script python to make a training
"""
import os
from baseutil import dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#change  the trainingset name  if necessary
trainingname = 'training-small.nc'

outdir = '../figures/examples'

#data directory
datadir = '../data'

ds = dataset(basename=os.path.join(datadir, trainingname))

#%%
#for unknown reasons, it seems to work better if dataset is instaciated
# before importing keras
from modelutil import get_model_4layers

# name of the neural networks
name = 'model_4layers'

X_train, X_test, yt_train, yt_test = train_test_split(ds.X, ds.yt, test_size=0.2)
X_test, X_valid, yt_test, yt_valid =  train_test_split(X_test, yt_test, test_size=0.5)

#%%
#dimension of input data
img_rows, img_cols, img_canal = X_train.shape[1:4]

make_model = get_model_4layers(img_rows, img_cols, img_canal)
print(make_model.summary())
#%%
history = make_model.fit(X_train, yt_train, validation_data=(X_valid,yt_valid), epochs=50,batch_size=10,shuffle=True)
print(history.history.keys())

make_model.save(os.path.join(datadir, name))

fig = plt.figure()
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],'r',label='test')
title = 'Loss_'
plt.title('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()
plt.suptitle(title)
fig.savefig(os.path.join(outdir,title+name+'.png'))
