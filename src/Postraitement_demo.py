# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:30:04 2018
@author: kouassi
"""

import os
import xarray as xr
import numpy as np
from evalutil import power_spectrum, test_rmse, test_corrcoef, test_varratio
import matplotlib.pyplot as plt

# CHOIX DU DATASET A TRAITER
AllInputDs = ["cl1", "sq1", "sq2"] # cl1 : cloud ; sq1: nauge carré ; sq2 : nuage carré centré
inputds = AllInputDs[0]

# TEST_SET_NAME & WEIGHTS BASENAME & DATA DIRECTORY
if (inputds == AllInputDs[0]):             ## cloud 
    InTestName =    'BaseTest_cloud.nc'
    weightsName = 'weights_cloud.nc'  
    OutTestName = 'DatasetNN_cloud.nc'
    rmseFileName = 'rmse_cloud.npy'
    corrcoefFileName = 'corrcoef_cloud.npy'
    varRatioFileName = 'varRatio_cloud.npy'
    # Data Directory     
    datadir = '../data/cloud'
    # Figure Directory
    outdir = '../figures/examples/BaseTest_Cloud'
    
elif (inputds == AllInputDs[1]):           ## plusieurs carrés 
    InTestName =  'BaseTest_square1.nc'
    OutTestName = 'datasetNN_Square1.nc'
    weightsName = 'weights_square1.nc'
    rmseFileName = 'rmse_square1.npy'
    corrcoefFileName = 'corrcoef_square1.npy'
    varRatioFileName = 'varRatio_square1.npy'
    # Data Directory
    datadir = '../data/Square1'
    # Figure Directory
    outdir = '../figures/examples/BaseTest_Square1'
    
elif (inputds == AllInputDs[2]):           ## square 2 : nuage 'carré' centré
    TestName =    'BaseTest_Square2.nc'
    weightsName = 'weights_square2.nc'
    OutTestName = 'datasetNN_Square2.nc'
    rmseFileName = 'rmse_square2.npy'
    corrcoefFileName = 'corrcoef_square2.npy'
    varRatioFileName = 'varRatio_square2.npy'
    # Data Directory
    datadir =   '../data/Square2'
    modelsdir = '../data/models'
    # Figure Directory 
    outdir = '../figures/examples/BaseTest_Square2'
else :
    #raise valueError("le code du dataset n'existe pas, choississez le bon ou créer un nouveau")
    print("le code du dataset n'existe pas, choississez le bon ou créer un nouveau")

# LECTURE DES DATASETS
inputTestName = os.path.join(datadir,InTestName)
outputTestName = os.path.join(os.path.join(datadir,OutTestName))
inputTest = xr.open_dataset(inputTestName)
outputTest = xr.open_dataset(outputTestName)

# Extraction des valeurs 
chla_final = outputTest['yfinal']
chla_final = chla_final.values.squeeze() # Supression du canal rajouté pour le reseau de neurones
chla_true = outputTest['yt']
chla_true = chla_true.values.squeeze()   # Supression du canal rajouté pour le reseau de neurones
Amask = np.array(inputTest.amask, dtype = int)

# %% RMSE & R2 & Corrcoef & Rapport de variance

index = np.array(outputTest.index).tolist() # Index des images 
# initialisation des listes 
RMSE = []; CORRCOEF = []; mevalCrop = []; VARRATIO = []
VTM = [] ; VPM =[]
for j,ind in enumerate(index):
    ytrue = outputTest.yt[j].values.squeeze()
    if ytrue.shape[0]>2:
        ytrue = ytrue.squeeze()
    ypred = outputTest.ypredict[j].values
    if ypred.shape[0]>2:
        ypred = ypred.squeeze()
    amask = inputTest.amask[j].values
    nmask = inputTest.nmask[j].values
    # Calcul du rmse par imagette (sur chaque carré imputé)
    ev_rmse = test_rmse(ytrue, ypred, amask=amask, nmask=nmask, dx=2, dy=0, coefC=0.1, coefN=1, cbool=True, cwidth=16, cheight=16 )
    RMSE.append(ev_rmse)
    # Calcul du coef de correlation (sur chaque carré imputé)
    corr_coef = test_corrcoef(ytrue,ypred)
    CORRCOEF.append(corr_coef)
    # Calcul du rapport de variance
    varratio, vpm, vtm = test_varratio(ytrue, ypred, bool1=False)
    VTM.append(vtm); VPM.append(vpm)
    VARRATIO.append(varratio)
    
np.save(os.path.join(datadir,rmseFileName), np.array(RMSE))
np.save(os.path.join(datadir,corrcoefFileName), np.array(CORRCOEF))
np.save(os.path.join(datadir,varRatioFileName), np.array(VARRATIO))

# %% Calcul du spectre de puissance et Visulaisation
PSD2D, PSD1D = power_spectrum(inputTestName, outputTestName, cb=True, plot = True)    

# %% Visualisation images de réference : % Author : arimoux %

index = [27,66,85,87,88,107,112,113]
# parameters for figures
SAVE = False
for i,ind in enumerate(index):
    vmin, vmax = -1,0       # -1.5,1
    plot_full=True
    plot_square=True
    plot_hist = True
    
    nanval = -1e5
    isNotSquare = np.equal(outputTest.yt[ind,:,:],nanval)
    yt0 = (1-isNotSquare)*outputTest.yt[ind,:,:]
    yt0 = yt0.values
    yp0 = (1-isNotSquare)*outputTest.ypredict[ind,:,:]
    yp0 = yp0.values
    xi,yi,zi = np.nonzero(yt0)
    yts = yt0[xi[0]:xi[-1],yi[0]:yi[-1]]
    yps = yp0[xi[0]:xi[-1],yi[0]:yi[-1]]
    
    yt_full = isNotSquare*outputTest.X[ind] + (1-isNotSquare)*outputTest.yt[ind]
    
    
    if plot_full:
        fig, axes= plt.subplots(ncols=3)
        im0=axes[0].imshow(np.log10(outputTest.X[ind].squeeze()),vmin=vmin,vmax=vmax)
        axes[0].set_title("Input image",fontsize=14)
        im1=axes[1].imshow(np.log10(yt_full.squeeze()),vmin=vmin,vmax=vmax)
        axes[1].set_title("True Image",fontsize=14)
        im2=axes[2].imshow(np.log10(outputTest.yfinal[ind].squeeze()),vmin=vmin,vmax=vmax)
        axes[2].set_title("Inpainted Image",fontsize=14)
        cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.07])
        cb=fig.colorbar(im1, cax=cbar_ax,orientation='horizontal')
        cb.set_label("log10 of Chlorophyll a concentration [$mg/m^3$]",fontsize=14)
    
        filename = 'Image_' + str(int(outputTest.index[ind]))+'_full'
        title = 'Full images comparison'
        plt.suptitle(title,fontsize=16)
        if SAVE:
            plt.savefig(os.path.join(outdir,filename+'.png'))
    
    if plot_square:
        fig, axes= plt.subplots(ncols=2)
        im0=axes[0].imshow(np.log10(yts.squeeze()),vmin=vmin,vmax=vmax)
        axes[0].set_title("True image",fontsize=14)
        im1=axes[1].imshow(np.log10(yps.squeeze()),vmin=vmin,vmax=vmax)
        axes[1].set_title("Inpainted Image",fontsize=14)
        cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.07])
        cb=fig.colorbar(im1, cax=cbar_ax,orientation='horizontal')
        cb.set_label("log10 of Chlorophyll a concentration [$mg/m^3$]",fontsize=14)
    
        filename = 'Image_' + str(int(outputTest.index[ind]))+'_square'
        title = 'Comparison of missing square'
        plt.suptitle(title,fontsize=16)
        if SAVE:
            plt.savefig(os.path.join(outdir,filename+'.png'))

