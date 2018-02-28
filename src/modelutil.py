#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:26 2018

@author: jbrlod
"""

from keras import losses
import keras.backend as K



def masked_mse(y_true,y_pred):
    nanval = -1e5
    isMask = K.equal(y_true,nanval)
    isMask = 1 - K.cast(isMask,dtype=K.floatx())
    y_true = y_true*isMask
    y_pred = y_pred*isMask
    return (losses.mean_squared_error(y_true,y_pred))
    