#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:26 2018

@author: jbrlod
"""

from keras import losses
import keras.backend as K
from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose,Conv2D
from keras.layers.core import Activation
from keras.layers import MaxPooling2D, concatenate, Input



def get_model_2layers(img_rows,img_cols):
    #mettre kes inputs
    inputs = Input(shape=(img_rows, img_cols, 1))
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


def masked_mse(y_true,y_pred):
    nanval = -1e5
    isMask = K.equal(y_true,nanval)
    isMask = 1 - K.cast(isMask,dtype=K.floatx())
    y_true = y_true*isMask
    y_pred = y_pred*isMask
    return (losses.mean_squared_error(y_true,y_pred))
    