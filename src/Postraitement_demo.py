# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:30:04 2018
@author: kouassi
"""
import os
import shutil
import xarray as xr
import numpy as np
from evalutil import qqplot_2samples, power_spectrum, test_rmse, test_corrcoef, test_varratio, mask_apply_crop
import matplotlib.pyplot as plt
import sklearn

# Briques DU DATASET A TRAITER
PredPattern = ['cloudm']#,'squarem'] #,'cloud1','square1'];    # list of the cloud's patterns
AppPattern= ['cloudm']#,'squarem'];                            # List des modeles issus des bases d'apprentissage  
InbaseName = ['test','valid']
allalgo = ['cnn','cnnfc','cnnfclog','krig']#,'gan']            # lists of deep learning algorithms 
allLayers=['2layers','3layers','4layers', '5layers'];
allLossf = ['mmse','cmse']; 
EntryData=['classic','binary']#, 'temporal' ,'bintemp'] 

# Choix du dataset à traiter
predpattern = PredPattern[0]
appPattern = AppPattern[0]
basetype = InbaseName[1]
algo = allalgo[2]
lossf = allLossf[0]
bincl = EntryData[1]
nblayers = allLayers[2] # model characteristics

if (algo != allalgo[-1]):
    dataset_type = algo+nblayers+'_'+basetype+'_'+appPattern+'_'+predpattern+'_'+lossf+'_'+bincl
    OutName = dataset_type+'.nc'           # fichier en sortie de test_model.py
    # Valid or Test dataset name & WEIGHTS BASENAME & DATA DIRECTORY
    weightsName = 'weights_'+basetype+'_'+predpattern+'.nc'
    chlatmcFileName = 'chlatmc_'+dataset_type+'.npy'
    chlapmcFileName = 'chlapmc_'+dataset_type+'.npy'
    rmseFileName = 'rmse_'+dataset_type+'.npy'
    corrcoefFileName = 'corrcoef_'+dataset_type+'.npy'
    varRatioFileName = 'varRatio_'+dataset_type+'.npy'
    varTruemFileName = 'varTruem_'+dataset_type+'.npy'
    varPredmFileName = 'varPredm_'+dataset_type+'.npy'
    R2globalFileName = 'R2global_'+dataset_type+'.npy'
    RMSEglobalFileName = 'RMSEglobal_'+dataset_type+'.npy'
    CorrCoefglobalFileName = 'CorrCoefglobal_'+dataset_type+'.npy'
    # Data Directory  
    inputdatadir = '../data/'+predpattern+'/'+basetype                # input data path of the neural network
    datadir = '../data/'+appPattern+'/'+basetype+'/'+algo+'/'+nblayers+'/'+predpattern+'/'+bincl
        
    # Figure Directory
    figdir = '../figures/examples/'+appPattern+'/'+algo+'/'+nblayers+'/'+basetype+'/'+bincl
                     
if (algo == allalgo[-1]):                    # Toujours mettre le krigeage en dernière position sur la liste allalgo
    dataset_type = algo+'_'+basetype+'_'+predpattern
    OutName = dataset_type+'.nc'             # nom du fichier en sortie de l'algo de krigeage
    weightsName = 'weights_'+basetype+'_'+predpattern+'.nc'
    chlatmcFileName = 'chlatmc_'+dataset_type+'.npy'
    chlapmcFileName = 'chlapmc_'+dataset_type+'.npy'
    rmseFileName = 'rmse_'+dataset_type+'.npy'
    corrcoefFileName = 'corrcoef_'+dataset_type+'.npy'
    varRatioFileName = 'varRatio_'+dataset_type+'.npy'
    varTruemFileName = 'varTruem_'+dataset_type+'.npy'
    varPredmFileName = 'varPredm_'+dataset_type+'.npy'
    R2globalFileName = 'R2global_'+dataset_type+'.npy'
    RMSEglobalFileName = 'RMSEglobal_'+dataset_type+'.npy'
    CorrCoefglobalFileName = 'CorrCoefglobal_'+dataset_type+'.npy'
    # Data Directory 
    inputdatadir = '../data/'+predpattern+'/'+basetype  
    datadir = inputdatadir+'/'+algo
    # Figure Directory 
    figdir = '../figures/examples/'+predpattern+'/'+algo+'/'+basetype

# %% Traitement et sauvegarde des données

print(dataset_type)
# Creation des chemins vers les dossiers
if os.path.isdir(datadir):
    print('Directory already created')
else:
    shutil.rmtree(datadir,ignore_errors=True) 
    os.makedirs(datadir)    
# LECTURE DES DATASETS
outputName = os.path.join(datadir, OutName)
output = xr.open_dataset(outputName)
#    c = np.ones_like(output.values); c = 10*c
#    output.values = np.power(c, output.values)
    
index = np.array(output.index).tolist() # Index des images 
# initialisation des listes 
RMSE = []; CORRCOEF = []; VARRATIO = []; VARPM=[]; VARTM=[]
CHLAPMC = []; CHLATMC = []; 
for k,_ in enumerate(index):
    ytrue = output.yt[k].values
    if ytrue.shape[0]>2:
        ytrue = ytrue.squeeze() 
        
    yfinal = output.yfinal[k].values
    if yfinal.shape[0]>2:
        yfinal = yfinal.squeeze()
    
    amask = output.amask[k].values # modifier qd tous les output ds auront des amask et nmask en sortie
    
    #Calcul du rmse par imagette (sur chaque region completée)
    ev_rmse = test_rmse(ytrue, yfinal, amask=amask, dx=2, dy=0, cbool=True, cwidth=16, cheight=16)
    RMSE.append(ev_rmse)

    # Calcul du coef de correlation (sur chaque region completée)
    corr_coef = test_corrcoef(ytrue,yfinal,amask=amask)
    CORRCOEF.append(corr_coef)
    
    # Calcul du rapport de variance
    varratio, vpm, vtm = test_varratio(ytrue, yfinal,amask=amask, bool1=False)
    vpmx = np.argwhere(vpm==0); vtmx = np.argwhere(vtm==0)
    if vtmx.shape[0]>0 and np.array_equal(vpmx, vtmx): # Le rapport des variances nulles est égale à 1
        varratio=1.0
    VARRATIO.append(varratio)
    VARPM.append(vpm)
    VARTM.append(vtm)
#        
     #Récuperation des valeurs prédites (completées)
    _, _, ytr_mc, yfn_mc  = mask_apply_crop(ytrue, yfinal, amask=amask, cwidth=16, cheight=16, cb=True)
    if yfn_mc.shape[0]>800:
            CHLAPMC.extend(yfn_mc)
            CHLATMC.extend(ytr_mc)   
# Coefficient de determination R2 global                        
R2global = sklearn.metrics.r2_score(np.array(CHLATMC,dtype=float),np.array(CHLAPMC,dtype=float))
print('R2global : '+str(R2global))
RMSEglobal = sklearn.metrics.mean_squared_error(np.array(CHLATMC,dtype=float), np.array(CHLAPMC,dtype=float))
print('RMSEglobal : '+str(RMSEglobal))
# Coefficient de correlation global
CorrCoefglobal = np.corrcoef(np.array(CHLATMC,dtype=float),np.array(CHLAPMC,dtype=float))[0][1]
print('CorrCoefglobal : '+str(CorrCoefglobal))

SAVE=True
if SAVE:
    # Sauvegarde des données  
    np.save(os.path.join(datadir,chlatmcFileName), np.array(CHLATMC))
    np.save(os.path.join(datadir,chlapmcFileName), np.array(CHLAPMC))
    np.save(os.path.join(datadir,rmseFileName), np.array(RMSE))
    np.save(os.path.join(datadir,corrcoefFileName), np.array(CORRCOEF))
    np.save(os.path.join(datadir,varRatioFileName), np.array(VARRATIO))
    np.save(os.path.join(datadir,varPredmFileName), np.array(VARPM))
    np.save(os.path.join(datadir,varTruemFileName), np.array(VARTM))
    np.save(os.path.join(datadir,R2globalFileName), R2global)
    np.save(os.path.join(datadir,RMSEglobalFileName), RMSEglobal)
    np.save(os.path.join(datadir,CorrCoefglobalFileName), CorrCoefglobal)


# %% VISUALISATION DE RMSE, RAPPORT DE VARIANCE, VARIANCES REELLES & PREDITES
# Obligation de lancer la 1ère cellulle

SAVE = True

print(dataset_type)

if SAVE:
    if not os.path.isdir(figdir):
        shutil.rmtree(figdir, ignore_errors=True)
        os.makedirs(figdir)
        
CHLATMC = np.load(os.path.join(datadir,chlatmcFileName))
CHLAPMC = np.load(os.path.join(datadir,chlapmcFileName))
####  1°) Scatter plot: Chla Predict vs Chla True
plt.figure()
plt.scatter(np.log10(np.array(CHLATMC)), np.log10(np.array(CHLAPMC)), c='black')
plt.xlabel('Chla Réel',fontsize=14); plt.ylabel('Chla Prédit', fontsize=14)
plt.xlim(-1, 0.4); 
plt.ylim(-1.0, 0.4)
#plt.title('Chla predict vs Chla true', fontsize=14)
plt.plot(range(-2,2),range(-2,2),'-r',linewidth=2.5)
plt.savefig(os.path.join(figdir,'chlascatterplot_'+dataset_type+'.png'))

#### 2°) Histogramme des erreurs : écarts entre données prédites et réelles
plt.figure()
c = np.subtract(np.array(CHLAPMC),np.array(CHLATMC));
weights = np.ones_like(c)/float(len(c)) # normalisation
plt.hist(c, bins=100, weights=weights, alpha=0.75)
plt.xlim(-0.40, 0.40); #plt.ylim(-1.6, 0.2)
plt.xlabel('Error chlaPred-ChlaTrue',fontsize=14)
plt.ylabel('Frequence',fontsize=14)
plt.savefig(os.path.join(figdir,'errorhistogram_'+dataset_type+'.png'))
#    
### 4°) Histogramme du coefficient de corr.
corrcoef = np.load(os.path.join(datadir,corrcoefFileName))
ind = np.argwhere(corrcoef == None)
if (ind.shape[0]>0):
    mask = np.ones(corrcoef.shape[0], dtype=bool)
    mask[ind] = False
    corrcoef=corrcoef[mask]
corrcoef = np.array(corrcoef,dtype=float)
ind = np.isnan(corrcoef); ind = np.argwhere(ind==True)
if (ind.shape[0]>0):
    mask = np.ones(corrcoef.shape[0], dtype=bool)
    mask[ind] = False
    corrcoef=corrcoef[mask]
corrcoef = np.array(corrcoef,dtype=float)
plt.figure()
weights = np.ones_like(corrcoef)/float(len(corrcoef)) #normalisation
plt.hist(corrcoef,weights=weights, bins=20, alpha=0.75)
plt.xlabel('coefficient de corr.', fontsize=14)
plt.ylabel('Frequence', fontsize=14)
plt.savefig(os.path.join(figdir,'corrcoefhistogram_'+dataset_type+'.png'))
# analyse du resultat
idx = np.argwhere(corrcoef==np.max(corrcoef))
print(idx) 
#cnn : 33, 43, 44, 82 & krig : 82 pour corrcoef<0.3
#cnn : 44, 82 pour corrcoef<0.2
#cnn : 10,17,24,28,31,32,33,38,42,43,44,57,70,82,90,94 ;  krig:8,10,38,82 pour corrcoef<0.5
#cnn : 26 ; krig : 23  pour max(corrcoef)

### Scatter plot des variances et QQ plot
vpm = np.load(os.path.join(datadir,varPredmFileName))
vtm = np.load(os.path.join(datadir,varTruemFileName))
idx = np.argwhere(vpm == None)
if (idx.shape[0]>0):
    mask = np.ones(vpm.shape[0], dtype=bool)
    mask[idx] = False
    vpm=vpm[mask]; vtm=vtm[mask];
vpm = np.array(vpm,dtype=float); vtm = np.array(vtm,dtype=float)

# 6°) Scatter plot des variances
plt.figure()
plt.scatter(vtm,vpm, c='black')
plt.plot(range(0,2),range(0,2),'-r',linewidth=2.5)
plt.xlim(-0.001,0.035); plt.ylim(-0.001,0.035)
plt.xlabel('Variance de Chla true', fontsize=14)
plt.ylabel('Variance de Chla predite', fontsize=14)
plt.savefig(os.path.join(figdir,'varscatterplot_'+dataset_type+'.png'))

#    # 7°)  QQ plot des variances   
figname = os.path.join(figdir,'varqqplot_'+dataset_type+'.png')
fig = qqplot_2samples(vtm, vpm,figname, xlabel='variance vraie', ylabel='variance predite', line='45')

    
## 8°) Spectre de variance
outputName = os.path.join(datadir, OutName)
output = xr.open_dataset(outputName) ; 
amask = np.array(output.amask.values, dtype=float); 
notamask = np.array(np.logical_not(output.amask.values), dtype=float);
yfinal = output.yfinal.values.squeeze() ; yt = output.yt.values.squeeze()
CHLATRFULL = np.add(np.multiply(amask,yt),np.multiply(notamask,yfinal))
#CHLATRFULL=output.ytfull.values
PSD2D,PSD2D_True, PSD1D, PSD1D_True = power_spectrum(outputName,CHLATRFULL=CHLATRFULL, cb=True,plot =False)
chla_final = output.yfinal.values.squeeze()
    
plot=True
if plot == True :
    # Visualisation d'une image tirée aléatoirement 
    nim = 1
    #idx = np.random.randint(0,chla_final.shape[0], nim)
    idx = 82
    #for i,ind in enumerate(idx):
    for ind in range(idx,idx+1):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8 ])
        ax.set_title("Inpainted Image",fontsize=14)
        i = ax.imshow(np.log10(chla_final[ind,:,:]),vmin = -0.8, vmax=0.2)
        colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8 ])
        fig.colorbar(i, cax=colorbar_ax)
        fig.savefig(os.path.join(figdir,'Inpainted_Image_'+dataset_type+'.png'))

        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title("True Image", fontsize=14)
        i = ax.imshow(np.log10(CHLATRFULL[ind,:,:]),vmin=-0.8, vmax=0.2)
        fig.colorbar(i)
        fig.savefig(os.path.join(figdir,'True_Image_'+dataset_type+'.png'))
        
#        fig, ax = plt.subplots()
#        ax.imshow(np.log10(PSD2D[ind]))
#        ax.set_title("Inpainted 2D Spectrum",fontsize=14)
#        
#        fig, ax = plt.subplots()
#        ax.imshow(np.log10(PSD2D_True[ind]))
#        ax.set_title("Inpainted 2D Spectrum",fontsize=14)

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.15, 0.65, 0.85 ])
        ax.plot(PSD1D[ind],label='Inpainted')
        ax.plot(PSD1D_True[ind],'r',label='True')
        ax.set_xscale('log', basex=2)
        ax.set_yscale('log', basey=2)
        ax.set_title("1D Spectrum",fontsize=14)
        ax.set_xlabel('Frequence Spatiale', fontsize=14);
        ax.set_ylabel("log2(Psuissance Spectrale)", fontsize=14);
        ax.legend(loc='upper right')
        fig.savefig(os.path.join(figdir,'radialspec1D_'+dataset_type+'.png'))
#%% Analyses des resultats

SAVE = True


if SAVE:
    if not os.path.isdir(figdir):
        shutil.rmtree(figdir, ignore_errors=True)
        os.makedirs(figdir)
        
#### 1°) Histogramme des RMSE
rmse = np.load(os.path.join(datadir,rmseFileName))
plt.figure()
plt.hist(rmse,bins=30, alpha=0.5)
plt.xlabel('RMSE de chaque region completée', fontsize=14)
plt.ylabel('Frequence', fontsize=14)
plt.savefig(os.path.join(figdir,'rmsehistogram_'+dataset_type+'.png'))
# Recherche des regions completées ayant des fort RMSE
#    ind = np.argwhere(rmse>0.002)
#    ind       

### RAPPORT DE VARIANCE
varratio = np.load(os.path.join(datadir,varRatioFileName))  
ind = np.argwhere(varratio == None)
if (ind.shape[0]>0):
    mask = np.ones(varratio.shape[0], dtype=bool)
    mask[ind] = False
    varratio=varratio[mask]
varratio = np.array(varratio,dtype=float)
ind1 = np.isnan(varratio); ind1 = np.argwhere(ind1==True)
if (ind1.shape[0]>0):
    mask = np.ones(varratio.shape[0], dtype=bool)
    mask[ind1] = False
    varratio=varratio[mask]
varratio = np.array(varratio,dtype=float)

vshape = varratio.shape
#### moyennes des rapports de variances
varratio_mean = np.mean(varratio)

#### les quartiles
q25 = np.percentile(varratio,q=25)
q50 = np.percentile(varratio,q=50)
q75 = np.percentile(varratio,q=75)
q95 = np.percentile(varratio,q=95)

### R2 global & RMSE global & Coefficient de correlation global
R2global = np.load(os.path.join(datadir,R2globalFileName)); 
RMSEglobal = np.load(os.path.join(datadir,RMSEglobalFileName))
CorrCoefglobal = np.load(os.path.join(datadir,CorrCoefglobalFileName))
print(dataset_type)
print('R2global = '+str(R2global))
print('RMSEglobal = '+str(RMSEglobal))
print('CorrCoefglobal= '+str(CorrCoefglobal))



#### 5°) Histogramme des rapports de variances
plt.figure()
plt.hist(varratio, bins=30, alpha=0.75)
plt.xlabel('Rapport de variance', fontsize=14)
plt.ylabel('Frequence', fontsize=14)
plt.savefig(os.path.join(figdir,'varRatiohistogram_'+dataset_type+'.png'))
#ind = np.argwhere(varratio>2)
#ind  

# %%  Résumé : Analyse comparative de plusieurs algo (CNN & krig)
# Chargement des données
varratio_cnn = np.load(os.path.join('../data/cloudm/valid/cnn/4layers/cloudm/binary','varRatio_cnn4layers_valid_cloudm_cloudm_mmse_binary.npy'))
varratio_cnnfc = np.load(os.path.join('../data/cloudm/valid/cnnfc/4layers/cloudm/classic','varRatio_cnnfc4layers_valid_cloudm_cloudm_mmse_classic.npy'))
varratio_cnnfclog = np.load(os.path.join('../data/cloudm/valid/cnnfclog/4layers/cloudm/binary','varRatio_cnnfclog4layers_valid_cloudm_cloudm_mmse_binary.npy'))
varratio_krig = np.load(os.path.join('../data/cloudm/valid/krig','varRatio_krig_valid_cloudm.npy'))

Varratio = np.empty(shape=(100,4))  
Varratio[:,0]=varratio_cnn;
Varratio[:,1]=varratio_cnnfc;
Varratio[:,2]=varratio_cnnfclog;
Varratio[:,3]=varratio_krig;
#### 6°) Boxplot des rapports de variances
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot(Varratio,1) 
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(['cnn4layers','cnn4layers_fc','cnn4layers_fc_log','krigeage'], rotation=10, fontsize=14) 
ax.set_ylabel('Ratio de variance', fontsize=14)
fig.savefig(os.path.join('../figures/examples/cloudm','boxplotVarratio.png'))
#plt.close(fig)

#### 7°) Scatter plot ratio var cnn vs krig
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Varratio[:,-1],Varratio[:,0])
ax.set_ylabel('ratio variance : cnn', fontsize=14)
ax.set_xlabel('ratio variance : krig.', fontsize=14)
ax.plot(range(0,3),range(0,3),'-r',linewidth=2.5)
fig.savefig(os.path.join('../figures/examples/cloudm','scatterplotVarratioCnnkrig_.png'))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Varratio[:,-1],Varratio[:,1])
ax.set_ylabel('ratio variance : cnn fully c.', fontsize=14)
ax.set_xlabel('ratio variance : krig.', fontsize=14)
ax.plot(range(0,4),range(0,4),'-r',linewidth=2.5)
fig.savefig(os.path.join('../figures/examples/cloudm','scatterplotVarratioCnnfckrig_.png'))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Varratio[:,-1],Varratio[:,2])
ax.set_ylabel('ratio variance : cnn fully c. log', fontsize=14)
ax.set_xlabel('ratio variance : krig.', fontsize=14)
ax.plot(range(0,4),range(0,4),'-r',linewidth=2.5)
fig.savefig(os.path.join('../figures/examples/cloudm','scatterplotVarratioCnnfclogkrig_.png'))
