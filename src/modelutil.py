#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:26 2018
@author: jbrlod
modified by A.Rimoux & M.Kouassi
"""

from keras import losses
import keras.backend as K
from keras.models import Model
from keras.layers.convolutional import Conv2DTranspose,Conv2D
from keras.layers.core import Activation
from keras.layers import MaxPooling2D, concatenate, Input
from sklearn.metrics import mean_squared_error as rmse1
import numpy as np

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

    model.compile(optimizer='adadelta', loss=masked_mse)

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

def mask_apply(y_true, y_pred):
    """ Application du masque pour travailler uniquement sur 
    le carré imputé par le réseau de neurone profond.
    y_true_m et y_pred_m sont uniquement le carré ou la forme imputée 
    par le reseau de neurone profond. 
    y_true et y_pred sont le carré ou la forme imputée 
    par le reseau de neurone profond , entouré par zero pour le reste de l'imagette."""
    nanval = -1e5
    # Réduction de dimension un array 2d (64*64)
    y_true.shape ; y_true = np.squeeze(y_true)  
    y_pred.shape; y_pred = np.squeeze(y_pred)
    # Définition et binarisation du masque
    isMask = np.empty_like(y_true)
    isMask[y_true == nanval] = 0
    isMask[y_true != nanval] = 1
    # Application du masque sur les y_true et y_pred
    y_true = y_true * isMask;
    y_pred = y_pred * isMask;
    y_true_m = y_true[y_true != 0]
    y_pred_m = y_pred[y_true != 0]
    return y_true, y_pred, y_true_m, y_pred_m

def mask_apply_crop(y_true, y_pred, cwidth, cheight, cb):
    """ Cette fonction permet de selectionner les ytrue et ypred qu'on 
    veut selon la position dans l'image du carré dans l'image.
    cheight : est la hauteur à prendre ou pas  en compte.
    cwidth :  est la largeur à prendre ou pas en compte.
    cb : booléen : if [True : crop extérieur] et [False : crop intérieur]"""
    nanval = -1e5
    # Définition et binarisation du masque
    isMask = np.empty_like(y_true)
    isMask[y_true == nanval] = 0
    isMask[y_true != nanval] = 1
    isMask = np.squeeze(isMask) 
    # Reduction de dimension de (64*64*1) à (64*64)
    isMask = np.squeeze(isMask) 
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    # Mask de selection des y true à plotter
    if (cb == True):
        bigmask = np.zeros(shape=(64,64))
        bigmask[cwidth:-cwidth, cheight:-cheight] = 1  
        isMask1 = isMask * bigmask
    elif (cb == False):
        bigmask = np.ones(shape=(64,64))
        bigmask[cwidth:-cwidth, cheight:-cheight] = 0  
        isMask1 = isMask * bigmask   
    # Application du masque sur 
    y_true = y_true * isMask1
    y_pred = y_pred * isMask1
    y_true_m = y_true[y_true != 0]
    y_pred_m = y_pred[y_true != 0]
    return y_true, y_pred, y_true_m, y_pred_m

def test_masked_mse(y_true,y_pred):
    """calcul du loss par root mean square (sklearn)"""
    ytr, yp, y_tr_m, y_pr_m = mask_apply(y_true,y_pred)
    mseLoss = rmse1(ytr,yp)
    return mseLoss

def test_masked_corrcoef(y_true,y_pred):
    """calcul du coefficient de correlation entre chla predict et chla true 
    du carré imputé par le reseau de neurones profond. """
    ytr, yp, y_tr_m, y_pr_m = mask_apply(y_true,y_pred)
    CorrCoef = np.corrcoef(y_tr_m, y_pr_m, bias=True)[0][1]
    return CorrCoef

def test_pixel_masked_loss(y_true, y_pred):
    ytrue, ypred, ytrue_m, ypred_m = mask_apply(y_true,y_pred)
    i = np.where(ytrue != 0)
    index_dim = np.shape(i)
    i_center = int(index_dim[1]/2) # indice de l'indice central du mask
    b = np.subtract(ytrue, ypred)
    # pixel central
    ix_center = i[0][i_center]
    iy_center = i[1][i_center]
    chla_center = b[ix_center,iy_center]

    # pixel central supérieur
    ix_up = i[0][0]
    chla_upcenter = b[ix_up,iy_center]

    # pixel central inférieur
    ix_down = i[0][-1]
    chla_DC = b[ix_down,iy_center]

    # pixel central gauche
    iy_left = i[1][0]
    chla_LC = b[ix_center,iy_left]

    # pixel central droit
    iy_right = i[1][-1]
    chla_RC = b[ix_center,iy_right]

    # pixel  des coins 
    chla_UL = b[ix_up,iy_left]       # coin supérieur gauche
    chla_UR = b[ix_up,iy_right]      # coin supérieur droit
    chla_DL = b[ix_down, iy_left]    # coin inférieur gauche
    chla_DR = b[ix_down, iy_right]   # coin inférieur droit 
    
    return chla_center, chla_UL, chla_UR, chla_DL, chla_DR
