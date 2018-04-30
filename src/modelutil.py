#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:26 2018
@author: jbrlod
modified by A.Rimoux & M.Kouassi
"""

from keras import losses
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose,Conv2D
from keras.layers.core import Activation
from keras.layers import MaxPooling2D, concatenate, Input

def get_model_4layers(img_rows,img_cols):
    #mettre les inputs
    inputs = Input(shape=(img_rows, img_cols, 1))
    #convolution classique 1
    conv_1 = Conv2D(16, (7, 7), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    #pooling 64->32
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    #convolution classique 2
    conv_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    #pooling 32->16
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    #convolution classique 3
    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    #pooling 16->8
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    #convolution classique 4
    conv_4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pl_3)
    act_4 = Activation('relu')(conv_4)
    #pooling 8->4
    pl_4=MaxPooling2D((2, 2), strides=(2, 2))(act_4)   
    #deconvolution classique 1
    deconv_1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(pl_4)
    dact_1 = Activation('relu')(deconv_1)
    #ajouter en input de la couche d'entée
    merge_1 = concatenate([dact_1, act_4], axis=3)   
    #refaire une convolution avec les deux informations  
    deconv_2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(merge_1)
    dact_2 = Activation('relu')(deconv_2)
    #ajouter en input de la couche d'entée
    merge_2 = concatenate([dact_2, act_3], axis=3)   
    #refaire une convolution avec les deux informations  
    deconv_3 = Conv2DTranspose(32 ,(3, 3), strides=(2, 2), padding='same')(merge_2)
    dact_3 = Activation('relu')(deconv_3)
    #ajouter en input de la couche d'entée
    merge_3 = concatenate([dact_3, act_2], axis=3)   
    #refaire une convolution avec les deux informations  
    deconv_4 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(merge_3)
    dact_4 = Activation('relu')(deconv_4)
    #ajouter en input de la couche d'entée
    merge_4 = concatenate([dact_4, inputs], axis=3) 
    #refaire une convolution avec les deux informations    
    final = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(merge_4)
    dact_5 = Activation('relu')(final)

    model = Model(inputs=[inputs], outputs=[dact_5])

    model.compile(optimizer='adadelta', loss=context_mse)

    return model

def get_model_3layers(img_rows,img_cols):
    #mettre les inputs
    inputs = Input(shape=(img_rows, img_cols, 1))
    #convolution classique 1
    conv_1 = Conv2D(16, (7, 7), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)
    #pooling 64->32
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    #convolution classique 2
    conv_2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(pl_1)
    act_2 = Activation('relu')(conv_2)
    #pooling 32->16
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    #convolution classique 3
    conv_3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(pl_2)
    act_3 = Activation('relu')(conv_3)
    #pooling 16->8
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    #deconvolution classique 1
    deconv_1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(pl_3)
    dact_1 = Activation('relu')(deconv_1)
    #ajouter en input de la couche d'entée
    merge_1 = concatenate([dact_1, act_3], axis=3)   
    #refaire une convolution avec les deux informations  
    deconv_2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(merge_1)
    dact_2 = Activation('relu')(deconv_2)
    #ajouter en input de la couche d'entée
    merge_2 = concatenate([dact_2, act_2], axis=3)   
    #refaire une convolution avec les deux informations  
    deconv_3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(merge_2)
    dact_3 = Activation('relu')(deconv_3)
    #ajouter en input de la couche d'entée
    merge_3 = concatenate([dact_3, inputs], axis=3)   
    #refaire une convolution avec les deux informations    
    final = Conv2D(1, (3, 3), strides=(1, 1), padding='same')(merge_3)
    dact_4 = Activation('relu')(final)

    model = Model(inputs=[inputs], outputs=[dact_4])

    model.compile(optimizer='adadelta', loss=masked_mse)

    return model


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
    return losses.mean_squared_error(y_true,y_pred)

def context_mse(y_true,y_pred):
    def loss_w(y_true, y_pred):
        chla_true, weights = tf.split(y_true,2,3)
        chla_pred1 = y_pred
        nanval = -1e5
        isMask = K.equal(chla_true,nanval)
        isMask = 1 - K.cast(isMask,dtype=K.floatx())
        chla_true = chla_true*isMask
        chla_pred1 = chla_pred1*isMask
        sq = K.square(tf.subtract(chla_pred1,chla_true))* weights
        mse = K.mean(sq)
        return mse
    return loss_w(y_true,y_pred)


