#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:58:49 2018

@author: arimoux
"""
import os
from keras.models import Model
from keras.optimizers import Adam
from modelutil_gan import model_autoencoder, model_discriminator, build_gan_2, gan_targets_2, masked_mse
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from keras.callbacks import CSVLogger
from keras.layers import Input

class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = model_discriminator()
        self.discriminator.name = 'model_discriminator'
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = model_autoencoder()
        self.generator.load_weights(os.path.join(load_path,"generator_pretrained_weights"))
        self.generator.name = 'model_generator'
        self.generator.compile(optimizer=optimizer,loss=masked_mse)

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = self.generator.input
        gen_missing = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(masked_img , [gen_missing, valid])
        self.combined.compile(loss=[masked_mse, 'binary_crossentropy'],
                              loss_weights=[0.999, 0.001],
                              optimizer=optimizer)

    def train_generator(self,epochs=20,batch_size=20):

        optimizer_G = Adam(0.002)
        self.generator.compile(loss=masked_mse,
            optimizer=optimizer_G,
            metrics=['accuracy'])
        self.generator.summary()

        x_train, x_test, y_train, y_test, y_input_train, y_input_test = train_test_split(Input_autoencoder, Target_autoencoder, Input_ytfull, test_size=0.1)
        x_test, x_valid, y_test, y_valid, y_input_test, y_input_valid = train_test_split(x_test, y_test, y_input_test, test_size=0.5)

        csv_logger = CSVLogger((os.path.join(save_path,'gen_log.csv')), append=True, separator=';')
        g_loss = self.generator.fit(x_train, y_train, validation_data=(x_valid,y_valid), epochs=epochs, batch_size=batch_size, shuffle=True,callbacks=[csv_logger])

        self.generator.save_weights(os.path.join(save_path,"generator_modif_fit_weights"))

    def train_gan(self, epochs, batch_size=128, save_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        self.combined.summary()
        loss = np.zeros([epochs,5],dtype=float)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            true_imgs = y_input_train[idx]

            # Generate a batch of new images
            gen_imgs = self.generator.predict(imgs)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(true_imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            imgs_label = y_train[idx]

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(imgs, [imgs_label, valid_y])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            loss[epoch,:] = [epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]]
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

        np.savetxt(os.path.join(save_path,"loss_gan.csv"), loss, delimiter=";")

    def save_imgs(self, epoch):
        number_i = 2
        idx = np.random.randint(0, x_train.shape[0],number_i)
        imgs = x_train[idx]
        true_imgs = y_input_train[idx]
        gen_imgs = self.generator.predict(imgs)
        isCloud = np.where(y_train[idx]==nanval,True,False)
        gen_imgs_final = isCloud*imgs + ~isCloud*gen_imgs

        fig, axs = plt.subplots(number_i,4)
        cnt = 0
        for i in range(number_i):
            vmax = true_imgs[cnt,:,:,0].max()
            vmin = 0
            axs[i,0].imshow(imgs[cnt,:,:,0],vmin=vmin,vmax=vmax)
            axs[i,1].imshow(true_imgs[cnt,:,:,0],vmin=vmin,vmax=vmax)
            axs[i,2].imshow(gen_imgs[cnt,:,:,0],vmin=vmin,vmax=vmax)
            axs[i,3].imshow(gen_imgs_final[cnt,:,:,0],vmin=vmin,vmax=vmax)

            cnt += 1
        fig.savefig(os.path.join(save_path,"gan%d.png" % epoch))

        false_imgs = self.discriminator.predict(gen_imgs_final)
        true_imgs = self.discriminator.predict(true_imgs)
        discri_pred = np.concatenate((false_imgs,true_imgs),axis=1)
        np.savetxt(os.path.join(save_path,"pred_discri_%d.txt" % epoch),discri_pred)

    def save_models(self):
        self.generator.save_weights(os.path.join(save_path,".gan_generator_weights"))
        self.discriminator.save_weights(os.path.join(save_path,"gan_discriminator_weights"))

    def test(self,x_test,y_test,y_input_test):
        ypredict = self.generator.predict(x_test)
        ##generate a combinated image of original & predicted images
        isCloud = np.where(y_test==nanval,True,False)
        yfinal = isCloud*x_test + ~isCloud*ypredict
        return yfinal


if __name__ == '__main__':

    path = '/net/argos/data/parvati/arimoux/share/GAN'
    load_path = '/net/argos/data/parvati/arimoux/share/GAN' # Path for pre-trained generator model
    save_path = '/net/argos/data/parvati/arimoux/share/GAN/gan_train' # Path to save directory

    # data loading
    nanval=-1e5
    ds1 = xr.open_dataset(os.path.join(path,'base_training_GAN_64.nc'))
    Input_autoencoder = ds1.X.fillna(0).expand_dims('canal',3)
    Target_autoencoder = ds1.yt.fillna(nanval).expand_dims('canal',3)
    Input_ytfull = ds1.ytfull.fillna(0).expand_dims('canal',3)
    x_train, x_test, y_train, y_test, y_input_train, y_input_test = train_test_split(Input_autoencoder, Target_autoencoder, Input_ytfull, test_size=0.1)
    x_test, x_valid, y_test, y_valid, y_input_test, y_input_valid = train_test_split(x_test, y_test, y_input_test, test_size=0.5)

    # completion after gan training
    gan = GAN()

    # Generator training
#    gan.train_generator(epochs=30,batch_size=20)

    # completion before gan training
    y_final_before = gan.test(x_test,y_test,y_input_test)

    # Gan training
    gan.train_gan(epochs=5000, batch_size=32, save_interval=100)

    # Models saving
    gan.save_models()

    # completion after gan training
    y_final_after = gan.test(x_test,y_test,y_input_test)

    #save predictions in .nc file
    dsout = xr.Dataset({'X':(['index','y','x'],x_test[:,:,:,0]),
                             'ygenerator':(['index','y','x'],y_final_before[:,:,:,0]),
                             'ytfull':(['index','y','x'],y_input_test[:,:,:,0]),
                             'ygan':(['index','y','x'],y_final_after[:,:,:,0])})
    dsout.to_netcdf(os.path.join(save_path,'dataset_test_gan.nc'))