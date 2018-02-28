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

name = 'model_7layers'
#ds = xr.open_dataset('../data/training.nc')
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
    conv_1 = Conv2D(20, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    #pooling 64->32
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    #convolution classique2
    conv_2 = Conv2D(15, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    #pooling 32->16
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    #convolution classique3
    conv_3 = Conv2D(10, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    #pooling 16->8
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    #deconvolution classique
    deconv_1 = Conv2DTranspose(10, (3, 3), strides=(2, 2), padding='same')(pl_3)
    dact_1 = Activation('relu')(deconv_1)
    #ajouter en input de la couche d'entée
    merge_1 = concatenate([dact_1, act_3], axis=3)   
    #refaire une convolution avec les deux informations  
    deconv_2 = Conv2DTranspose(15, (3, 3), strides=(2, 2), padding='same')(merge_1)
    dact_2 = Activation('relu')(deconv_2)
    #ajouter en input de la couche d'entée
    merge_2 = concatenate([dact_2, act_2], axis=3)   
    #refaire une convolution avec les deux informations  
    deconv_3 = Conv2DTranspose(25, (3, 3), strides=(2, 2), padding='same')(merge_2)
    dact_3 = Activation('relu')(deconv_3)
    #ajouter en input de la couche d'entée
    merge_3 = concatenate([dact_3, inputs], axis=3)   
    #refaire une convolution avec les deux informations    
    final = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(merge_3)
    dact_4 = Activation('relu')(final)

    model = Model(inputs=[inputs], outputs=[dact_4])

    model.compile(optimizer='adadelta', loss=masked_mse)

    return model


model = get_model()
model.fit(ds.X,ds.yt,epochs=20)
#
model.save(name)

