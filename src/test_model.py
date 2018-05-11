#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:31:46 2018
script python to plot some outputs of the model
@author: jbrlod
modified by mkouassi & arimoux
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import xarray as xr
from baseutil import dataset, weights_mask

alltestname = ['BaseTest_Cloud1.nc','BaseTest_Clouds.nc','BaseTest_Square1.nc','BaseTest_Squares.nc']
#alltestname = ['base_validation_1cloud.nc', 'base_validation_multiple_clouds.nc', 'base_validation_1square.nc','base_validation_multiple_squares.nc']
## DATA DIRECTORY
AllInputDs = ["cl1", "cl+", "sq1", "sq+"]              # cl1 : cloud ; sq1: nauge carré ; sq2 : nuage carré centré
allPattern = ['cloud1','clouds','square1','squares'];  # list of the cloud's patterns
alldir = ['cloud1','clouds','square1','squares'];      # list of all data directories
AllDeepModels = ['cnn','gan']                          # lists of deep learning algorithms 
deepmodels=AllDeepModels[0]
count = 0;
for icode, inputds in enumerate(AllInputDs):
    if (inputds==AllInputDs[0]):
        ddir = alldir[0]
        datadir = '../data/cloud/'+ddir
        pattern = allPattern[0]
        testname = alltestname[0] 
        weightsName = 'Weights_Cloud1.nc'
        
    elif (inputds == AllInputDs[1]):
        ddir = alldir[1]
        datadir = '../data/cloud/'+ddir 
        pattern = allPattern[1]
        testname = alltestname[1] 
        weightsName = 'Weights_Clouds.nc'
        
    elif (inputds == AllInputDs[2]):
        ddir = alldir[2]
        datadir = '../data/square/'+ddir
        pattern = allPattern[2]
        testname = alltestname[2]
        weightsName = 'Weights_Square1.nc'
        
    elif (inputds == AllInputDs[3]):
        ddir = alldir[3]
        datadir = '../data/square/'+ddir
        pattern = allPattern[3]
        testname = alltestname[3] 
        weightsName = 'Weights_Squares.nc'
        
    
    # CALL THE DATASET CLASS    
    ds = dataset(basename=os.path.join(datadir,testname))
    
    # BUILDING of Binary input mask  and loss function weight masks 
    Inputds = os.path.join(datadir,testname)
    weightdsName = os.path.join(datadir,weightsName)
    wds = weights_mask(Inputds, weightdsName, weight_c=0.1, weight_n=1)
    weight = wds.weights.expand_dims('canal',3)
    Xbinary = wds.bmask.expand_dims('canal',3)
    ## Concatenation des bases des dataArray 'yt',weights
    yt_2dim = xr.concat((ds.yt, weight),dim='canal')
        
    ## MODEL DIRECTORY
    from keras.models import load_model
    from modelutil import context_mse, masked_mse
    
    allLoss = ['mmse','cmse']; 
    allLayers=['2layers','3layers','4layers']; nbLayers = allLayers[2]
    # list of models directories based on the inputs data models
    allmdir=['classic','binary']#, 'temporal' ,'bintemp'] 
    for mcode, mdir in enumerate(allmdir):
        modelsdir = '../data/models/'+deepmodels+'/'+mdir
        # 
        if (mdir == allmdir[0]):
            Xbinary.values = np.array(Xbinary.values, dtype='int')
        elif (mdir == allmdir[1]):
            Xbinary.values = np.ones_like(Xbinary.values)
            
        X_2dim = xr.concat((ds.X, Xbinary),dim='canal')
        ## Splitting 
        #X_test, X_valid, yt_test, yt_valid = train_test_split(X_2dim, yt_2dim, test_size=0.2)
            
        # Choice of the loss function
        for mcode, Lossf in enumerate(allLoss):
            # Name of the neural architecture
            name = 'model_'+nbLayers+'_'+pattern+'_'+Lossf+'_'+mdir
            # Outputname
            outname = 'DatasetNN_'+nbLayers+'_'+pattern+'_'+Lossf+'_'+mdir+'_'+deepmodels+'.nc'
            # Model loading
            if Lossf == allLoss[0]:
                model = load_model(os.path.join(modelsdir,name), custom_objects={'masked_mse':masked_mse})
            elif Lossf == allLoss[1]:
                model = load_model(os.path.join(modelsdir,name),custom_objects={'context_mse':context_mse})
            else:
                raise ValueError("Choisir une fonction de cout existante ou en créer une dans modelutil")
        
            
            ypredict = xr.DataArray(model.predict(X_2dim,batch_size=10,verbose=1),coords=ds.yt.coords)
            
            # Generate a combinated image of original & predicted images
            isCloud = np.equal(ds.X,0)
            yfinal = (1-isCloud)*ds.X + isCloud*ypredict
            
            # Save prediction
            dsout = xr.Dataset({'X':ds.X,'yt':ds.yt,'ypredict':ypredict,'yfinal':yfinal,'ytfull':ds.ytfull})
            dsout.to_netcdf(os.path.join(datadir,outname))
            
            print(count); count +=1
            
            # Plot some random images
            PLOT = False
            
            #save the images
            SAVE = False
            
            if PLOT:   
                
                nim = 20 #number of images to be plot
                ii = np.random.randint(0,dsout.index.size,nim)
                vmin, vmax = -1.5,0
            
                for i,ind in enumerate(ii):
                    fig, axes= plt.subplots(ncols=4)
                    axes[0].imshow(np.log10(dsout.X[ind].squeeze()),vmin=vmin,vmax=vmax)
                    axes[1].imshow(np.log10(dsout.yt[ind].squeeze()),vmin=vmin,vmax=vmax)
                    axes[2].imshow(np.log10(dsout.ypredict[ind].squeeze()),vmin=vmin,vmax=vmax)
                    axes[3].imshow(np.log10(dsout.yfinal[ind].squeeze()),vmin=vmin,vmax=vmax)
                    title = 'Image_' + str(int(dsout.index[ind]))
                    plt.suptitle(title)
                    if SAVE: 
                        # example dir : A FAIRE +++ 
                        exampledir = os.path.join('../figures/examples',os.path.splitext(name)[0])
                        if os.path.isdir():
                            print('Directory already created')
                            plt.savefig(os.path.join(exampledir,title+'.png'))  
                        else:
                            shutil.rmtree(exampledir,ignore_errors=True) 
                            os.makedirs(exampledir)
                            plt.savefig(os.path.join(exampledir,title+'.png'))
                                
                                