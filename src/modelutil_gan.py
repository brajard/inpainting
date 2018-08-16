#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:35:26 2018
@author: jbrlod
modified by A.Rimoux & M.Kouassi
"""




import tensorflow as tf
import keras.backend as K
from keras import losses
from keras.layers import Lambda, Dense, Flatten, Input, concatenate, InputLayer #,Reshape, merge
from keras.models import  Model #,Sequential
from keras.layers import MaxPooling2D 
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2DTranspose, Conv2D, Cropping2D
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np


#def model_generator(latent_dim, input_shape, hidden_dim=512, reg=lambda: l1l2(1e-7, 0)):
#    inputs = Input(shape=(input_shape[0], input_shape[1], 1))
#    model=Model(inputs,inputs,name="generator")
#    return model


def model_discriminator(img_rows=64,img_cols=64, img_canal=1,filter_number=32,kernel_size=(4,4),activation='sigmoid',optimizer='adam',padding='same'):
    
    inputs_d = Input(shape=(img_rows,img_cols, img_canal),name='input_1d')
    cropped_1 = Cropping2D(cropping=((16, 16), (16, 16)),name='cropped_1_d')(inputs_d)
#    conv_1_d = Conv2D(32, (4, 4), strides=(1, 1), padding='same',name='conv_1_d')(cropped_1)
#    act_1_d = Activation('relu',name='act_1_d')(conv_1_d)
#    pl_1_d=MaxPooling2D((2, 2), strides=(2, 2),name='pl_1_d')(act_1_d)
#    conv_2_d = Conv2D(64, (4, 4), strides=(1, 1), padding='same',name='conv_2_d')(pl_1_d)
#    act_2_d = Activation('relu',name='act_2_d')(conv_2_d)
#    pl_2_d=MaxPooling2D((2, 2), strides=(2, 2),name='pl_2_d')(act_2_d)
#    conv_3_d = Conv2D(128, (4, 4), strides=(1, 1), padding='same',name='conv_3_d')(pl_2_d)
#    act_3_d = Activation('relu',name='act_3_d')(conv_3_d)
#    pl_3_d=MaxPooling2D((2, 2), strides=(2, 2),name='pl_3_d')(act_3_d)
#    fc_1_d=Flatten(name='fc_1_d')(pl_3_d)
#    fc_2_d=Dense(40,name='fc_2_d')(fc_1_d)
#    act_4_d=Activation('relu',name='act_4_d')(fc_2_d)
#    fc_3_d=Dense(25,name='fc_3_d')(act_4_d)
#    act_5_d=Activation('relu',name='act_5_d')(fc_3_d)
#    fc_4_d=Dense(10,name='fc_4_d')(act_5_d)
#    act_6_d=Activation('relu',name='act_6_d')(fc_4_d)
#    fc_5_d=Dense(1,name='fc_5_d')(act_6_d)
#    act_7_d=Activation('sigmoid',name='act_7_d')(fc_5_d)
#    return Model(inputs_d,act_7_d)

    conv_1_d = Conv2D(filter_number, kernel_size, strides=(1, 1), padding=padding,name='conv_1_d')(cropped_1)
    #act_1_d = Activation('relu',name='act_1_d')(conv_1_d)
    act_1_d = LeakyReLU()(conv_1_d)    
    #pooling 32->16
    pl_1_d=MaxPooling2D((2, 2), strides=(2, 2),name='pl_1_d')(act_1_d)
    #convolution classique 2
    conv_2_d = Conv2D(filter_number*2, kernel_size, strides=(1, 1), padding=padding,name='conv_2_d')(pl_1_d)
    #act_2_d = Activation('relu',name='act_2_d')(conv_2_d)
    act_2_d = LeakyReLU()(conv_2_d)    
    #pooling 16->8
    pl_2_d=MaxPooling2D((2, 2), strides=(2, 2),name='pl_2_d')(act_2_d)
    #convolution classique 3
    conv_3_d = Conv2D(filter_number*4, kernel_size, strides=(1, 1), padding=padding,name='conv_3_d')(pl_2_d)
#    act_3_d = Activation('relu',name='act_3_d')(conv_3_d)
    act_3_d = LeakyReLU()(conv_3_d)    
    #pooling 8->4
    pl_3_d=MaxPooling2D((2, 2), strides=(2, 2),name='pl_3_d')(act_3_d)
    #convolution classique 4
    conv_4_d = Conv2D(filter_number*8, kernel_size, strides=(1, 1), padding=padding,name='conv_4_d')(pl_3_d)
#    act_4_d = Activation('relu',name='act_4_d')(conv_4_d)
    act_4_d = LeakyReLU()(conv_4_d)    
    #pooling 4->2
    pl_4_d=MaxPooling2D((2, 2), strides=(2, 2),name='pl_4_d')(act_4_d)   

    fc_1_d=Flatten(name='fc_1_d')(pl_4_d)
    fc_2_d=Dense(1,name='fc_2_d')(fc_1_d)
    act_6_d=Activation('sigmoid',name='act_6_d')(fc_2_d)
    return Model(inputs_d,act_6_d)



def model_autoencoder(img_rows=64,img_cols=64,img_canal=1,filter_number=32,kernel_size=(4,4),activation='linear',optimizer='adam',padding='same'):
    #mettre les inputs
    inputs = Input(shape=(img_rows, img_cols, img_canal))
    #convolution classique 1
    conv_1 = Conv2D(filter_number, kernel_size, strides=(1, 1), padding=padding)(inputs)
    act_1 = Activation('relu')(conv_1)
    #pooling 64->32
    pl_1=MaxPooling2D((2, 2), strides=(2, 2))(act_1)
    #convolution classique 2
    conv_2 = Conv2D(filter_number*2, kernel_size, strides=(1, 1), padding=padding)(pl_1)
    act_2 = Activation('relu')(conv_2)
    #pooling 32->16
    pl_2=MaxPooling2D((2, 2), strides=(2, 2))(act_2)
    #convolution classique 3
    conv_3 = Conv2D(filter_number*4, kernel_size, strides=(1, 1), padding=padding)(pl_2)
    act_3 = Activation('relu')(conv_3)
    #pooling 16->8
    pl_3=MaxPooling2D((2, 2), strides=(2, 2))(act_3)
    #convolution classique 4
    conv_4 = Conv2D(filter_number*8, kernel_size, strides=(1, 1), padding=padding)(pl_3)
    act_4 = Activation('relu')(conv_4)
    #pooling 8->4
    pl_4=MaxPooling2D((2, 2), strides=(2, 2))(act_4)   
    #Fully-connected layer
    bottleneck=Dense(16, activation='relu')(pl_4)
     #deconvolution classique 1
    layer_up_1=Lambda(lambda images: tf.image.resize_images(bottleneck,[8,8],align_corners=True))(bottleneck)
    deconv_1 = Conv2D(filter_number*8, kernel_size, strides=(1,1), padding=padding)(layer_up_1)
    dact_1 = Activation('relu')(deconv_1)
    #ajouter en input de la couche d'entée
    merge_1 = concatenate([dact_1, act_4], axis=3)   
    #refaire une convolution avec les deux informations  
    #layer_pad_1=InputLayer(input_tensor=tf.pad(merge1, paddings, "CONSTANT")
    layer_up_2=Lambda(lambda images: tf.image.resize_images(merge_1,[16,16],align_corners=True))(merge_1)
    deconv_2 = Conv2DTranspose(filter_number*4, kernel_size, strides=(1, 1), padding=padding)(layer_up_2)
    dact_2 = Activation('relu')(deconv_2)
    #ajouter en input de la couche d'entée
    merge_2 = concatenate([dact_2, act_3], axis=3)   
    #refaire une convolution avec les deux informations  
    layer_up_3=Lambda(lambda images: tf.image.resize_images(merge_2,[32,32],align_corners=True))(merge_2)
    deconv_3 = Conv2D(filter_number*2 ,kernel_size, strides=(1,1), padding=padding)(layer_up_3)
    dact_3 = Activation('relu')(deconv_3)
    #ajouter en input de la couche d'entée
    merge_3 = concatenate([dact_3, act_2], axis=3)   # Avec Skip connection
    #Refaire une convolution avec les deux informations  
    layer_up_4=Lambda(lambda images: tf.image.resize_images(merge_3,[64,64],align_corners=True))(merge_3)
    deconv_4 = Conv2D(filter_number, kernel_size, strides=(1,1), padding=padding)(layer_up_4)
    dact_4 = Activation('relu')(deconv_4)
    #ajouter en input de la couche d'entée
    merge_4 = concatenate([dact_4, inputs], axis=3) 
    #refaire une convolution avec les deux informations    
    final_1 = Conv2D(1, kernel_size, strides=(1, 1), padding=padding)(merge_4)
    
    dact_5 = Activation(activation)(final_1)
    model = Model(inputs=[inputs], outputs=[dact_5])
    return model

def penalized_loss(x,weight_hole=0.1,weight_ol=1):
    def loss(y_true, y_pred):
        zeronan = 0
        isMask_ol = nearby_hole(x,6)
        isMask_ol = K.cast(isMask_ol,dtype=K.floatx())
        isMask = K.equal(x,zeronan) # mask in the region with hole (integer)
        isMask_square = K.cast(isMask,dtype=K.floatx()) # mask for the pixel where the hole is
        isMask_out = 1 - isMask_square # mask  for the pixels not considering the hole
        loss_square = losses.mean_squared_error(y_true*isMask_square,y_pred*isMask_square)
        #loss_out = losses.mean_squared_error(y_true*isMask_out,y_pred*isMask_out)
        loss_ol = losses.mean_squared_error(y_true*isMask_ol,y_pred*isMask_ol)
        return loss_square*weight_hole  + loss_ol*weight_ol #  loss outputs sum
    return loss

#def masked_mse(y_true,y_pred):
#    nanval = 0
#    isMask = nearby_hole(y_true,y_pred,4)
#    isMask = 1 - K.cast(isMask,dtype=K.floatx())
#    y_true = y_true*isMask
#    y_pred = y_pred*isMask
#    return (losses.mean_squared_error(y_true,y_pred))


def masked_mse(y_true,y_pred):
    def classic_loss(y_true,y_pred):
        nanval = -1e5
        #chla_true, _ = tf.split(y_true,2,3)
        chla_true = y_true
        isMask = K.equal(y_true,nanval)
        isMask = 1 - K.cast(isMask,dtype=K.floatx())
        chla_true = chla_true*isMask
        y_pred = y_pred*isMask
        return losses.mean_squared_error(chla_true,y_pred)
    return classic_loss(y_true,y_pred)

def build_gan(generator, discriminator, name="gan"):
    """
    Build GAN from generator and discriminator
    Model is (z, x) -> (yfake, yreal)
    :param generator: Model (z -> x)
    :param discriminator: Model (x -> y)
    :return: GAN model
    """
    xfake = Activation("linear", name="xfake")(generator(generator.inputs))
    yfake = Activation("linear", name="yfake")(discriminator(generator(generator.inputs)))
    yreal = Activation("linear", name="yreal")(discriminator(discriminator.inputs))
    model = Model(generator.inputs + discriminator.inputs, [yreal, yfake, xfake], name=name)
    return model


def gan_targets(n,ytrain):
    """
    Standard training targets
    [generator_fake, generator_real, discriminator_fake, discriminator_real] = [1, 0, 0, 1]
    :param n: number of samples
    :return: array of targets
    """
    generator_yfake = np.ones((n, 1))
    generator_yreal = np.zeros((n, 1))
    discriminator_yfake = np.zeros((n, 1))
    discriminator_yreal = np.ones((n, 1))
    generator_xfake = ytrain
    discriminator_xfake = ytrain 
    return [generator_yreal, generator_yfake, generator_xfake, discriminator_yreal, discriminator_yfake, discriminator_xfake]

def build_gan_2(generator, discriminator, name="gan"):
    """
    Build GAN from generator and discriminator
    Model is (z, x) -> (yfake, yreal)
    :param generator: Model (z -> x)
    :param discriminator: Model (x -> y)
    :return: GAN model
    """
    yfake = Activation("linear", name="yfake")(discriminator(generator(generator.inputs)))
    yreal = Activation("linear", name="yreal")(discriminator(discriminator.inputs))
    model = Model(generator.inputs + discriminator.inputs, [yreal, yfake], name=name)
    return model


def gan_targets_2(n):
    """
    Standard training targets
    [generator_fake, generator_real, discriminator_fake, discriminator_real] = [1, 0, 0, 1]
    :param n: number of samples
    :return: array of targets
    """
    generator_yfake = np.ones((n, 1))
    generator_yreal = np.zeros((n, 1))
    discriminator_yfake = np.zeros((n, 1))
    discriminator_yreal = np.ones((n, 1))
    return [generator_yreal, generator_yfake, discriminator_yreal, discriminator_yfake]