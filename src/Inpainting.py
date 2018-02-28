#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from baseutil import dataset
#change le path of the training set if necessary
ds = dataset(basename='../data/trainingset.nc')
#for some reasons, it seems to work better if dataset is instaciated
#before importing keras

from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose,Conv2D
from keras.layers.core import Activation
from keras.layers import MaxPooling2D, concatenate, Input
from modelutil import masked_mse
#import xarray as xr

name = 'model_2layers'
#ds = xr.open_dataset('../data/trainingset.nc')
#X = ds['X'].values #.values facultatif, c'est pour avoir un np.array
#yt = ds['yt'].values

#dimension des données d'entrée
img_rows, img_cols = 64,64
n_feat_in, filter_size, filter_size = 25 , 3, 3


##tester avec 1 puis implermenter avec plus de phases internes de conv et deconv
def get_model():
    #mettre kes inputs
    inputs = Input(shape=(64, 64, 1))
    #convolution classique
    conv_1 = Conv2D(25, (11, 11), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    #pooling
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    #deconvolution classique
    deconv_1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(pl_1)
    dact_1 = Activation('relu')(deconv_1)
    #ajouter en input de la couche d'entée
    merge_1 = concatenate([dact_1, inputs], axis=3)
    #refaire une convolution avec les deux informations
    final = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(merge_1)
    dact_2 = Activation('relu')(final)

    model = Model(inputs=[inputs], outputs=[dact_2])

    model.compile(optimizer='adadelta', loss=masked_mse)

    return model


model = get_model()
model.fit(ds.X,ds.yt,epochs=20)

model.save(name)

