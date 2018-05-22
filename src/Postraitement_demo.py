# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:30:04 2018
@author: kouassi
"""
import os
import shutil
import xarray as xr
import numpy as np
from evalutil import power_spectrum, test_rmse, test_corrcoef, test_varratio, mask_apply_crop
import matplotlib.pyplot as plt
import sklearn

# CHOIX DU DATASET A TRAITER
AllInputDs = ["cl1", "cl+", "sq1", "sq+"]             # cl1 : cloud ; sq1: nauge carré ; sq2 : nuage carré centré
allPattern = ['cloud1','square1','cloudm','squarem']; # list of the cloud's patterns
alldir= ['cloudm','squarem'];                         # List des modeles issus des bases d'apprentissage  
InbaseName = ['test','valid']
allalgo = ['krig','cnn']#,'gan',]            # lists of deep learning algorithms 
allLayers=['2layers','3layers','4layers'];
allLossf = ['mmse','cmse']; 
EntryData=['classic','binary']#, 'temporal' ,'bintemp'] 

OutName=[]; InName=[]; datadir=[]; outdir=[]; R2globalFileName=[]; Dataset_type=[];  RMSEglobalFileName=[] # Initialsation de la liste des noms des fichiers
weightsName = []; rmseFileName =[]; corrcoefFileName=[]; varTruemFileName=[]; varPredmFileName=[]; varRatioFileName=[];
chlatmcFileName=[]; chlapmcFileName = []
for i, inputds in enumerate(AllInputDs): 
    for j, algo in enumerate(allalgo):
        if (algo == allalgo[1]):
            for _, mdir in enumerate(alldir):
                for _, lossf in enumerate(allLossf):
                    for _, bincl in enumerate(EntryData):
                        dataset_type = allalgo[1]+allLayers[2]+'_'+InbaseName[1]+'_'+mdir+'_'+allPattern[i]+'_'+lossf+'_'+bincl
                        Dataset_type.append(dataset_type)
                        OutName.append(dataset_type+'.nc')
                        # Valid or Test dataset name & WEIGHTS BASENAME & DATA DIRECTORY
                        InName.append(InbaseName[1]+'_'+allPattern[i]+'.nc')
                        weightsName.append('weights_'+dataset_type+'.nc')
                        chlatmcFileName.append('chlatmc_'+dataset_type+'.npy')
                        chlapmcFileName.append('chlapmc_'+dataset_type+'.npy')
                        rmseFileName.append('rmse_'+dataset_type+'.npy')
                        corrcoefFileName.append('corrcoef_'+dataset_type+'.npy')
                        varRatioFileName.append('varRatio_'+dataset_type+'.npy')
                        varTruemFileName.append('varTruem_'+dataset_type+'.npy')
                        varPredmFileName.append('varPredm_'+dataset_type+'.npy')
                        R2globalFileName.append('R2global_'+dataset_type+'.npy')
                        RMSEglobalFileName.append('RMSEglobalFileName_'+dataset_type+'.npy')
                        # Data Directory  
                        datadir.append('../data/'+mdir+'/'+InbaseName[1]+'/'+allalgo[1]+allLayers[2]+'/'+allPattern[i])
                        # Figure Directory
                        outdir.append('../figures/examples/'+mdir+'/'+InbaseName[1]+'/'+allalgo[1]+allLayers[2]+'/'+allPattern[i])
                             
            if (algo == allalgo[0]):
                dataset_type = allalgo[0]+'_'+InbaseName[1]+'_'+allPattern[i]
                Dataset_type.append(dataset_type)
                OutName.append(dataset_type+'.nc')
                # Valid or Test dataset name & WEIGHTS BASENAME & DATA DIRECTORY
                InName.append(allalgo[0]+'_'+InbaseName[1]+'_'+allPattern[i]+'.nc')
                weightsName.append('weights_'+dataset_type+'.nc')
                chlatmcFileName.append('chlatmc_'+dataset_type+'.npy')
                chlapmcFileName.append('chlapmc_'+dataset_type+'.npy')
                rmseFileName.append('rmse_'+dataset_type+'.npy')
                corrcoefFileName.append('corrcoef_'+dataset_type+'.npy')
                varRatioFileName.append('varRatio_'+dataset_type+'.npy')
                varTruemFileName.append('varTruem_'+dataset_type+'.npy')
                varPredmFileName.append('varPredm_'+dataset_type+'.npy')
                R2globalFileName.append('R2global_'+dataset_type+'.npy')
                RMSEglobalFileName.append('RMSEglobalFileName_'+dataset_type+'.npy')
                # Data Directory  
                datadir.append('../data/'+allPattern[i]+'/'+InbaseName[1]+'/'+allalgo[0])
                # Figure Directory 
                outdir.append('../figures/examples/'+allPattern[i]+'/'+InbaseName[1]+'/'+allalgo[0])

# %% Traitement et sauvegarde des données
for j, outname in enumerate(OutName):
    # Creation des chemins vers les dossiers
    if os.path.isdir(datadir[j]):
        print('Directory already created')
    else:
        shutil.rmtree(datadir[j],ignore_errors=True) 
        os.makedirs(datadir[j])    
    # LECTURE DES DATASETS
    outputName = os.path.join(datadir[j], outname)
    #inputName = os.path.join(datadir[j], InName[j])
    output = xr.open_dataset(outputName)
    #Input = xr.open_dataset(inputName)  
        
    # RMSE & R2 & Corrcoef & Rapport de variance
    index = np.array(output.index).tolist() # Index des images 
    # initialisation des listes 
    RMSE = []; CORRCOEF = []; VARRATIO = []; VARPM=[]; VARTM=[]
    CHLAPMC = []; CHLATMC = []
    for k, ind in enumerate(index):
        ytrue = output.yt[k].values.squeeze()
        if ytrue.shape[0]>2:
            ytrue = ytrue.squeeze()
        yfinal = output.yfinal[k].values
        if yfinal.shape[0]>2:
            yfinal = yfinal.squeeze()
        amask = output.amask[k].values
        nmask = output.nmask[k].values
        # Calcul du rmse par imagette (sur chaque region completée)
        ev_rmse = test_rmse(ytrue, yfinal, amask=amask, dx=2, dy=0, cbool=True, cwidth=16, cheight=16)
        RMSE.append(ev_rmse)

        # Calcul du coef de correlation (sur chaque region completée)
        corr_coef = test_corrcoef(ytrue,yfinal,amask=amask)
        CORRCOEF.append(corr_coef)
        # Calcul du rapport de variance
        varratio, vpm, vtm = test_varratio(ytrue, yfinal,amask=amask, bool1=False)
        VARRATIO.append(varratio)
        VARPM.append(vpm)
        VARTM.append(vtm)
        # Récuperation des valeurs prédites (completées)
        _, _, ytr_mc, ypr_mc  = mask_apply_crop(ytrue, yfinal, amask=amask, cwidth=16, cheight=16, cb=True)
        if ypr_mc.shape[0]>0:
                CHLAPMC.extend(ypr_mc)
                CHLATMC.extend(ytr_mc)
    
    # Coefficient de correlation R2 global                        
    R2global = sklearn.metrics.r2_score(np.array(CHLATMC,dtype=float),np.array(CHLAPMC,dtype=float))
    RMSEglobal = sklearn.metrics.mean_squared_error(np.array(CHLATMC,dtype=float), np.array(CHLAPMC,dtype=float))
    print(j)
    
    SAVE=True
    if SAVE:
        # Sauvegarde des données  
        np.save(os.path.join(datadir[j],chlatmcFileName[j]), np.array(CHLATMC))
        np.save(os.path.join(datadir[j],chlapmcFileName[j]), np.array(CHLAPMC))
        np.save(os.path.join(datadir[j],rmseFileName[j]), np.array(RMSE))
        np.save(os.path.join(datadir[j],corrcoefFileName[j]), np.array(CORRCOEF))
        np.save(os.path.join(datadir[j],varRatioFileName[j]), np.array(VARRATIO))
        np.save(os.path.join(datadir[j],varPredmFileName[j]), np.array(VARPM))
        np.save(os.path.join(datadir[j],varTruemFileName[j]), np.array(VARTM))
        np.save(os.path.join(datadir[j],R2globalFileName[j]), R2global)
        np.save(os.path.join(datadir[j],RMSEglobalFileName[j]), RMSEglobal)

# %% VISUALISATION DE RMSE, RAPPORT DE VARIANCE, VARIANCES REELLES & PREDITES
# Obligation de lancer la 1ère cellulle

SAVE = True

for  j, outname in enumerate(OutName):
    if SAVE:
        if not os.path.isdir(outdir[j]):
            shutil.rmtree(outdir[j], ignore_errors=True)
            os.makedirs(outdir[j])
            
    CHLATMC = np.load(os.path.join(datadir[j],chlatmcFileName[j]))
    CHLAPMC = np.load(os.path.join(datadir[j],chlapmcFileName[j]))
    ####  1°) Scatter plot: Chla Predict vs Chla True
    plt.figure()
    plt.scatter(np.log10(np.array(CHLATMC)), np.log10(np.array(CHLAPMC)), c='black')
    plt.xlabel('Chla Réel',fontsize=14); plt.ylabel('Chla Prédit', fontsize=14)
    plt.xlim(-2, 0.7); plt.ylim(-3, 0.4)
    plt.title('Chla predict vs Chla true', fontsize=14)
    plt.plot(range(-2,2),range(-2,2),'-r',linewidth=2.5)
    plt.savefig(os.path.join(outdir[j],'chlascatterplot_'+Dataset_type[j]+'.png'))
    
    #### 2°) Histogramme des erreurs : écarts entre données prédites et réelles
    plt.figure()
    c = np.subtract(np.array(CHLAPMC),np.array(CHLATMC));
    #plt.hist(c, bins=np.arange(-0.1,0.1,0.003), alpha=0.5)
    plt.hist(c, bins=300, alpha=0.5)
    plt.xlim(-0.50, 0.50); #plt.ylim(-1.6, 0.2)
    plt.xlabel('Error chlaPred-ChlaTrue',fontsize=14)
    plt.ylabel('Frequency',fontsize=14)
    plt.savefig(os.path.join(outdir[j],'errorhistogram_'+Dataset_type[j]+'.png'))
    
    #### 3°) Histogramme des RMSE
    rmse = np.load(os.path.join(datadir[j],rmseFileName[j]))
    plt.figure()
    plt.hist(rmse,bins=30, alpha=0.5)
    plt.xlabel('RMSE de chaque region completée', fontsize=14)
    plt.ylabel('Frequence', fontsize=14)
    plt.savefig(os.path.join(outdir[j],'rmsehistogram_'+Dataset_type[j]+'.png'))
#    # Recherche des regions completées ayant des fort RMSE
#    #ind = np.argwhere(rmse>0.0003)
#    
    #### 4°) Histogramme du coefficient de corr.
    corrcoef = np.load(os.path.join(datadir[j],corrcoefFileName[j]))
    ind = np.argwhere(corrcoef == None)
    if (ind.shape[0]>0):
        mask = np.ones(corrcoef.shape[0], dtype=bool)
        mask[ind] = False
        corrcoef=corrcoef[mask]
    corrcoef = np.array(corrcoef,dtype=float)
    ind = np.isnan(corrcoef)
    if (ind.shape[0]>0):
        mask = np.ones(corrcoef.shape[0], dtype=bool)
        mask[ind] = False
        corrcoef=corrcoef[mask]
    corrcoef = np.array(corrcoef,dtype=float)
    plt.figure()
    plt.hist(corrcoef,bins=20, alpha=0.75)
    plt.xlabel('coefficient de corr.', fontsize=14)
    plt.ylabel('Frequence', fontsize=14)
    plt.savefig(os.path.join(outdir[j],'corrcoefhistogram_'+Dataset_type[j]+'.png'))
     
    #### 5°) Histogramme des rapports de variances
    varratio = np.load(os.path.join(datadir[j],varRatioFileName[j]))
    ind = np.argwhere(varratio == None)
    if (ind.shape[0]>0):
        mask = np.ones(varratio.shape[0], dtype=bool)
        mask[ind] = False
        varratio=varratio[mask]
    varratio = np.array(varratio,dtype=float)
    ind = np.isnan(varratio)
    if (ind.shape[0]>0):
        mask = np.ones(varratio.shape[0], dtype=bool)
        mask[ind] = False
        varratio=varratio[mask]
    varratio = np.array(varratio,dtype=float)
    plt.figure()
    plt.hist(varratio, bins=30, alpha=0.75)
    plt.xlabel('Rapport de variance', fontsize=14)
    plt.ylabel('Frequence', fontsize=14)
    #plt.xlim(-0.001,17)
    plt.savefig(os.path.join(outdir[j],'varRatiohistogram_'+Dataset_type[j]+'.png'))
    
    ## 6°) Scatter plot des variances
    vpm = np.load(os.path.join(datadir[j],varPredmFileName[j]))
    vtm = np.load(os.path.join(datadir[j],varTruemFileName[j]))
    plt.figure()
    plt.scatter(vtm,vpm, c='black')
    plt.plot(range(0,2),range(0,2),'-r',linewidth=2.5)
    plt.xlim(-0.005,0.08); plt.ylim(-0.01,0.08)
    plt.xlabel('Variance de Chla true', fontsize=14)
    plt.ylabel('Variance de Chla predite', fontsize=14)
    plt.savefig(os.path.join(outdir[j],'varscatterplot_'+Dataset_type[j]+'.png'))
    
    ## 6°) Spectre de variance
    PSD2D,PSD2D_True, PSD1D,PSD1D_True = power_spectrum(outputName,CHLATRFULL=output.ytfull.values, cb=True,plot = True)
    

# %% Visualisation images de réference
#
#index = [27,66,85,87,88,107,112,113] # index des images à visualiser 
## parameters for figures
#SAVE = False
#plot_full=True
#plot_square=True
#plot_hist = True
#vmin, vmax = -1,0
#
#CHLAPC, CHLATC, chla_pred, CHLATRFULL = inpainted_region(outputName, y1='ypred',nanval=-1e5, forspec=False)
#
#for i,ind in enumerate(index):  
#    if plot_full:
#        fig, axes= plt.subplots(ncols=3)
#        im0=axes[0].imshow(np.log10(output.X[ind].squeeze()),vmin=vmin,vmax=vmax)
#        axes[0].set_title("Input image",fontsize=14)
#        im1=axes[1].imshow(np.log10( CHLATRFULL[ind,:,:]),vmin=vmin,vmax=vmax)
#        axes[1].set_title("True Image",fontsize=14)
#        im2=axes[2].imshow(np.log10(output.yfinal[ind].values.squeeze()),vmin=vmin,vmax=vmax)
#        axes[2].set_title("Inpainted Image",fontsize=14)
#        cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.07])
#        cb=fig.colorbar(im1, cax=cbar_ax,orientation='horizontal')
#        cb.set_label("log10 of Chlorophyll a concentration [$mg/m^3$]",fontsize=14)
#    
#        filename = 'Image_' + str(int(output.index[ind]))+'_full'
#        title = 'Full images comparison'
#        plt.suptitle(title,fontsize=16)
#        if SAVE:
#            plt.savefig(os.path.join(outdir,filename+'.png'))
#    
#    if plot_square:
#        fig, axes= plt.subplots(ncols=2)
#        im0=axes[0].imshow(np.log10(np.array(CHLATC[ind])),vmin=vmin,vmax=vmax)
#        axes[0].set_title("True image",fontsize=14)
#        im1=axes[1].imshow(np.log10(np.array(CHLAPC[ind])),vmin=vmin,vmax=vmax)
#        axes[1].set_title("Inpainted Image",fontsize=14)
#        cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.07])
#        cb=fig.colorbar(im1, cax=cbar_ax,orientation='horizontal')
#        cb.set_label("log10 of Chlorophyll a concentration [$mg/m^3$]",fontsize=14)
#    
#        filename = 'Image_' + str(int(output.index[ind]))+'_square'
#        title = 'Comparison of missing square'
#        plt.suptitle(title,fontsize=16)
#        if SAVE:
#            plt.savefig(os.path.join(outdir,filename+'.png'))


