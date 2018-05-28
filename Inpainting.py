""" 
   Script python to make a training
"""
import os
import shutil
from baseutil import dataset, weights_mask
from sklearn.model_selection import train_test_split
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from modelutil import context_mse, masked_mse

# CHOIX DU DATASET A TRAITER
AllInputDs = ["cl1", "cl+", "sq1", "sq+"]   # cl1 : cloud ; sq1: nauge carré ; sq2 : nuage carré centré
allPatterns = ['cloud1','clouds','square1','squares']; 
# TRAINING_SET_NAME & WEIGHTS BASENAME & DATA DIRECTORY
count = 0
for icode, inputds in enumerate(AllInputDs):    
    if (inputds == AllInputDs[0]):             ## cloud 
        TrainingName = 'BaseTraining_Cloud1.nc'
        weightsName = 'Weights_Cloud1.nc'  
        pattern = allPatterns[0]
        # Data Directory     
        datadir = '../data/cloud/cloud1'
        # Figure Directory
        outdir = '../figures/examples/BaseTraining_Cloud/BaseTraining_Cloud1'
    
    elif (inputds == AllInputDs[1]): 
        TrainingName = 'Basetraining_clouds.nc'
        weightsName = 'Weights_Clouds.nc'  
        pattern = allPatterns[1]
        # Data Directory     
        datadir = '../data/cloud/clouds'
        # Figure Directory
        outdir = '../figures/examples/BaseTraining_Cloud/Basetraining_Clouds'
        
    elif (inputds == AllInputDs[2]):           ## square 1
        TrainingName = 'Basetraining_square1.nc'
        weightsName = 'Weights_Square1.nc'
        pattern = allPatterns[2]
        # Data Directory
        datadir = '../data/square/square1'
        # Figure Directory
        outdir = '../figures/examples/BaseTraining_Square/Basetraining_Square1'
        
    elif (inputds == AllInputDs[3]):           ## square 2 : nuage 'carré' centré
        TrainingName = 'BaseTraining_Squares.nc'
        weightsName = 'Weights_Squares.nc'
        pattern = allPatterns[3]
        # Data Directory
        datadir = '../data/square/squares'
        # Figure Directory 
        outdir = '../figures/examples/BaseTraining_Square/BaseTraining_Squares'
    else :
        raise ValueError("le code du dataset n'existe pas, choisissez le bon ou créer un nouveau")
        #print("le code du dataset n'existe pas, choississez le bon ou créer un nouveau.")
    
    
    # CALL THE DATASET CLASS
    ds = dataset(basename=os.path.join(datadir, TrainingName))
    
    # Binary input mask  and loss function weight masks BUILDING
    Inputds = os.path.join(datadir,TrainingName)
    weightdsName = os.path.join(datadir,weightsName)
    wds = weights_mask(Inputds, weightdsName, weight_c=0.1, weight_n=1)
    weight = wds.weights.expand_dims('canal',3)
    Xbinary = wds.bmask.expand_dims('canal',3)
    # CHOICE OF THE MODELS DIRECTORY
    AllDeepModels = ['cnn','gan'] ; # lists of deep learning algorithms 
    deepmodels=AllDeepModels[0]
    allmdir=['classic','binary']#, '/temporal' ,'/bintemp']; # list of models directories based on the inputs data models
    for mcode, mdir in enumerate(allmdir):
        modelsdir = '../data/models/'+deepmodels+'/'+mdir # model directory path
        if (mdir == allmdir[0]):
            Xbinary.values = np.array(Xbinary.values, dtype='int')
        elif (mdir == allmdir[1]):
            Xbinary.values = np.ones_like(Xbinary.values)
    
        ## Concatenation des bases des dataArray 'yt',weights
        #YT_2dim = np.stack([ds.yt, weight], axis = 3); YT_2dim = YT_2dim.squeeze()
        # OU 
        yt_2dim = xr.concat((ds.yt, weight),dim='canal')
        X_2dim = xr.concat((ds.X, Xbinary),dim='canal')
        # Splitting 
        X_train, X_valid, yt_train, yt_valid = train_test_split(X_2dim, yt_2dim, test_size=0.2)
        
        # For unknown reasons, it seems to work better if dataset is instaciated before importing keras.
        from modelutil import get_model_4layers1, get_model_4layers2
        
        ## NAME OF THE NEURAL NETWORK ARCHITECTURE 
        allLoss = ['mmse','cmse']; 
        allLayers=['2layers','3layers','4layers']; nbLayers = allLayers[2]
        
        for mcode, Lossf in enumerate(allLoss):
            name = 'model_'+nbLayers+'_'+pattern+'_'+Lossf+'_'+mdir
        
            # Dimensions of input data
            img_rows, img_cols, img_canal = X_2dim.shape[1:4]
            
            # Model loading and data training
            if (Lossf==allLoss[0]):
                model = get_model_4layers1(img_rows=img_rows, img_cols=img_cols, img_canal=img_canal)
            elif (Lossf==allLoss[1]):
                model = get_model_4layers2(img_rows=img_rows, img_cols=img_cols, img_canal=img_canal)
                
            #print(model.summary())
            history = model.fit(X_train, yt_train, validation_data=(X_valid,yt_valid), epochs=50, batch_size=10,shuffle=True)
            
            # Visualization of the loss function evolution
            print(history.history.keys())
            
            # model saving
            model.save(os.path.join(modelsdir,name))
            
            # training loss figure
            plt.figure()
            plt.plot(history.history['loss'],label='train')
            plt.plot(history.history['val_loss'],'r',label='test')
            title = 'Loss_'
            plt.title('loss')
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.show()
            plt.suptitle(title)
            
            # Sauvegarde du training loss
            
            if os.path.isdir(outdir):
                print('directory already created')
                plt.savefig(os.path.join(outdir,title+name+'.png'))  
            else:
                shutil.rmtree(outdir,ignore_errors=True)
                os.makedirs(outdir)
                plt.savefig(os.path.join(outdir,title+name+'.png'))
            
            print(count); count +=1

# %% TEST PHASE
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import xarray as xr
from baseutil import dataset, weights_mask

alltestname = ['BaseTest_Cloud1.nc','BaseTest_Clouds.nc','BaseTest_Square1.nc','BaseTest_Squares.nc']
#allvalidname = ['dataset_clouds_weights_bin_1cloud.nc', ]
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
                                
                                