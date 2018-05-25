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


# CHOIX DU DATASET A TRAITER
appPattern= ['cloudm']#,'squarem'];                                 # List des bases d'apprentissage  
allalgo = ['cnn']#,'cnnfc','gan']                                   # List of deep learning algorithms 
allLayers=['4layers']#,'5layers'];
allLossf = ['mmse','cmse']; 
EntryData=['classic','binary']#,'temporal','bintemp']             # list of models directories based on the inputs data models 

# TRAINING_SET_NAME & WEIGHTS BASENAME & DATA DIRECTORY
count=0;
for i, _ in enumerate(appPattern): 
    # input files for training
    trainingName = 'training_'+appPattern[i]+'.nc'
    weightsName = 'weights_'+appPattern[i]+'.nc'
    # Data Directory  
    trainingdir = '../data/'+appPattern[i]
    
    # Loading input dataset (training dataset)
    trainingds = os.path.join(trainingdir,trainingName)
    
    # CALL THE DATASET CLASS
    ds = dataset(basename=trainingds)
   
#    # BUILDING of Binary input mask  and loss function weight masks 
    weightdsName = os.path.join(trainingdir,weightsName)
    wds = weights_mask(trainingds, weightdsName, weight_c=0.1, weight_n=1)
    weight = wds.weights.expand_dims('canal',3)
#    #building of binary mask  for classic or binary entry
    Xbinary = wds.bmask.expand_dims('canal',3)
    for _, bincl in enumerate(EntryData): 
        if (bincl == EntryData[0]):
            Xbinary.values = np.ones_like(Xbinary.values) 
        elif (bincl == EntryData[1]):
            Xbinary.values = np.array(Xbinary.values, dtype='int')
    
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
        for _,algo in enumerate(allalgo):
            for _, nbLayers in enumerate(allLayers): 
                for _, lossf in enumerate(allLossf):
                    # Dimensions of input data
                    img_rows, img_cols, img_canal = X_2dim.shape[1:4]
                    # Model loading, model name, MODELS DIRECTORY path, output figures paths
                    if (lossf==allLossf[0]):
                        # Model loading
                        model = get_model_4layers1(img_rows=img_rows, img_cols=img_cols, img_canal=img_canal)
                        # Output filename
                        modelsName = 'model_'+algo+nbLayers+'_'+appPattern[i]+'_'+lossf+'_'+bincl
                        # Path of the Outputs model directory 
                        modelsdir = '../data/models/'+algo+'/'+nbLayers+'/'+bincl
                        # Figure Directory
                        figdir= '../figures/examples/'+appPattern[i]+'/'+algo+'/'+nbLayers+'/'+bincl
                    elif (lossf==allLossf[1]):
                        # Model loading
                        model = get_model_4layers2(img_rows=img_rows, img_cols=img_cols, img_canal=img_canal)
                        # Output filename
                        modelsName = 'model_'+algo+nbLayers+'_'+appPattern[i]+'_'+lossf+'_'+bincl
                        # Path of the Output model directory 
                        modelsdir = '../data/models/'+allalgo[0]+'/'+nbLayers+'/'+bincl
                        # Figure Directory
                        figdir= '../figures/examples/'+appPattern[i]+'/'+algo+'/'+nbLayers+'/'+'training'+'/'+bincl
                    
                    # Visualisation du model
                    #print(model.summary())
                    
                    # Data training
                    print('Training of the model : '+modelsName)
                    history = model.fit(X_train, yt_train, validation_data=(X_valid,yt_valid), epochs=80, batch_size=10, shuffle=True)
                    
                    # Visualization of the loss function evolution
                    print(history.history.keys())
                    
                    # Model saving
                    if os.path.isdir(modelsdir):
                        print('Model directory already created')
                    else:
                        shutil.rmtree(modelsdir,ignore_errors=True) 
                        os.makedirs(modelsdir) 
                    model.save(os.path.join(modelsdir,modelsName))
                    
                    # Training loss figure
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
                    if os.path.isdir(figdir):
                        print('figures directory already created')
                    else:
                        shutil.rmtree(figdir,ignore_errors=True)
                        os.makedirs(figdir)
                    plt.savefig(os.path.join(figdir,title+modelsName+'.png'))
                    
                    print(count); count +=1

