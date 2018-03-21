#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 16:27:48 2018
@author: kouassi
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from modelutil import test_masked_mse, test_pixel_masked_loss, mask_apply, mask_apply_crop, test_masked_corrcoef
from sklearn.metrics import mean_squared_error as rmse1
# inputname
inputname = '../data/dataset_nn.nc'
name = 'model_4layers'
exampledir = os.path.join('../figures/examples/',os.path.splitext(name)[0])

ds = xr.open_dataset(inputname)

# %% """ Erreurs (RMSE, ecart) et coeff. de correlation entre les données réelles ytrue et imputées par le réseau profond ypredict. """
# Autre manière de calculer l'erreur dans la zone imputée 
index = np.array(ds.index).tolist() # Index des images 
nb = np.array(ds.index).shape[0]    # Dsout.index.size
# initialisation des listes 
meval = []; chla_Corrcoef = [];
chlaC = []; chla_UL = []; chla_UL = []; 
chla_UR = []; chla_DR = []; chla_DL = [];

for j,ind1 in enumerate(index):
    ytrue1 = ds.yt[j].values
    ypred1 = ds.ypredict[j].values
    # Calcul du rmse par imagette (sur chaque carré imputé)
    ev_rmse = test_masked_mse(ytrue1,ypred1)
    meval.append(ev_rmse)
    # Calcul du coef de correlation (sur chaque carré imputé)
    corr_coef = test_masked_corrcoef(ytrue1,ypred1)
    chla_Corrcoef.append(corr_coef)
    # Ecart entre chlaTrue et chlaPred pour les pixels aux coins et au centre
    chlaCenter, chlaUL, chlaUR, chlaDL, chlaDR = test_pixel_masked_loss(ytrue1, ypred1)
    chlaC.append(chlaCenter); chla_UL.append(chlaUL); chla_UR.append(chlaUL);
    chla_DL.append(chlaDL); chla_DR.append(chlaDR);

# %% Visualisation de ces erreurs par Histogramme
    # Distribution de la RMSE sur le carré imputé
numBins = 60;
plt.figure('Erreur rmse')
plt.hist(meval, numBins)
plt.xlim(-0.00001,0.00045)
plt.xlabel("RMSE Values")
plt.ylabel("Frequency")
#plt.title('RMSE error on inpainted data')
plt.savefig('RMSE_error_distribution_on_inpainted_data.png')
#fig = plt.gcf()

# Figures scatter plot des erreurs entre ytrue et ypred sur le carré imputé 
num_bins = 10;
fig, axes = plt.subplots(ncols=5)
im0 = axes[0].hist(chlaC, num_bins)
axes[0].set_title("Central",fontsize=14)
axes[0].set_xlim(0,0.65); axes[0].set_ylim(0,301)
im1 = axes[1].hist(chla_UL, num_bins)
axes[1].set_title("Up Left",fontsize=14)
axes[1].set_xlim(0,0.65); axes[1].set_ylim(0,301)
im2 = axes[2].hist(chla_DL, num_bins)
axes[2].set_title("Down Left",fontsize=14)
axes[2].set_xlim(0,0.65); axes[3].set_ylim(0,301)
im3 = axes[3].hist(chla_UR, num_bins)
axes[3].set_title("Up Right",fontsize=14)
axes[3].set_xlim(0,0.65); axes[2].set_ylim(0,301)
im4 = axes[4].hist(chla_DR, num_bins)
axes[4].set_title("Down Right",fontsize=14)
axes[4].set_xlim(0,0.65); axes[4].set_ylim(0,301)
# %% Scatter plot de l'ensemble des données du carré imputé
chlaTrue_all = []; chlaPred_all = []; chlaTrue_All_Var = []; chlaPred_All_Var = [];
a = np.ones((8,8)) ; chlaT = np.empty_like(a)
for j,ind1 in enumerate(index):
    ytrue1 = ds.yt[j].values
    ypred1 = ds.ypredict[j].values
    yt,yp,ytr_m,ypr_m = mask_apply(ytrue1, ypred1)
    # list de tous les ytrue et ypred
    chlaTrue_all.extend(ytr_m.tolist())
    chlaPred_all.extend(ypr_m.tolist())
    # Calcul de la variance
    chlaTm = ytr_m ; chlaPm = ypr_m;
    vtm = np.var(chlaTm, dtype=np.float64, ddof=1)
    chlaTrue_All_Var.extend((np.ones(chlaTm.shape)*vtm).tolist())
    vpm = np.var(chlaPm, dtype=np.float64, ddof=1)
    chlaPred_All_Var.extend((np.ones(chlaPm.shape)*vpm).tolist())
# sauvegarde des données sous fichiers .npy
np.save('../data/chla_trueAll.npy', np.array(chlaTrue_all))
np.save('../data/chla_predAll.npy', np.array(chlaPred_all))
np.save('../data/chlaTrue_All_Var.npy', np.array(chlaTrue_All_Var))
np.save('../data/chlaPred_All_Var.npy', np.array(chlaPred_All_Var))

# Scatter plot: Chla Predict vs Chla True
plt.figure('Chla predict vs Chla true original')
plt.scatter(chlaTrue_all, chlaPred_all, c='black')
plt.xlabel('Chla Réel '); plt.ylabel('Chla Prédit')
plt.xlim(-0.1, 4.15); plt.ylim(0, 1.3)
#plt.title('Chla predict vs Chla true')
plt.plot(range(0,3),range(0,3),'-r',linewidth=2.5)
plt.savefig('../figures/examples/loss_evaluation/ytrue_vs_ypred_original.png')

# %% Scatter plot en fonction de la variances des valeurs de Chla de chaque carré
plt.figure()
cmhot = plt.cm.get_cmap("hot")
xs = chlaTrue_all
ys = chlaPred_all
zs = np.load('../data/chlaTrue_All_Var.npy')
plt.scatter(xs, ys, c=zs, cmap=cmhot)
plt.xlabel('chlaTrue',fontsize=14); plt.ylabel('chlaPred',fontsize=14)
plt.colorbar()
plt.show()

# Scatter plot échelle log10 : Chla Predict vs Chla True
plt.figure('Chla predict vs Chla true original')
plt.xlabel('Chla True'); plt.ylabel('Chla Predict')
plt.xlim(-1.6, 0.7); plt.ylim(-1.4, 0.2)
#plt.title('Chla predict vs Chla true')
plt.scatter(np.log10(chlaTrue_all), np.log10(chlaPred_all), c='black')
plt.plot(range(-2,2),range(-2,2),'-r',linewidth=2.5)
plt.savefig('../figures/examples/loss_evaluation/ytrue_vs_ypred_original_log10.png')

# Histogramme des écarts entre données prédites et réelles
plt.figure('histogramme des erreurs entre Chla pred et Chla true')
a= np.load('../data/chla_trueAll.npy'); a= np.log10(a);
b= np.load('../data/chla_predAll.npy'); b= np.log10(b);
c = np.subtract(b,a);
plt.hist(c, bins=np.arange(-0.4,0.4,0.02), alpha=0.5)
plt.xlabel('Error chlaPred-ChlaTrue')
plt.ylabel('Frequency')

# Calcul de rmse et coeff de correlation global sur les toutes données des carrés imputés
rmse_all = rmse1(chlaTrue_all, chlaPred_all)
CorrCoef_all = np.corrcoef(chlaTrue_all, chlaPred_all, bias=True)[0][1]

# %% Scatter plot des données du carré imputé et centré sur l'imagette 64x64

# Figures scatter plot ytrue vs ypred
chlaTrue_crop = []; chlaPred_crop =[];
for j,ind1 in enumerate(index):
    ytrue1 = ds.yt[j].values
    ypred1 = ds.ypredict[j].values
    cwidth = 16; cheight=16; cb = True;
    yt,yp,ytm,ypm = mask_apply_crop(ytrue1, ypred1, cwidth, cheight, cb)
    # liste des chla true et chla pred 
    chlaTrue_crop.extend(ytm.tolist())
    chlaPred_crop.extend(ypm.tolist())
    
# Sauvegarde des données
np.save('../data/chla_trueCrop.npy', np.array(chlaTrue_crop))
np.save('../data/chla_predCrop.npy', np.array(chlaPred_crop))

# Scatter plot: Chla Predict vs Chla True
plt.figure('Chla predict vs Chla true centré sur l''imagette')
plt.xlabel('Chla True', fontsize=14); plt.ylabel('Chla Predict',fontsize=14)
plt.xlim(0, 1.1); plt.ylim(0, 1.1)
#plt.title('Chla predict vs Chla true')
plt.scatter(chlaTrue_crop, chlaPred_crop, c='black')
plt.plot(range(0,3),range(0,3),'-r',linewidth=2.5)
plt.savefig('../figures/examples/loss_evaluation/ytrue_vs_ypred_32x32center.png')

# Chargement des données
a = np.load('../data/chla_trueCrop.npy'); a = np.log10(a);
b = np.load('../data/chla_predCrop.npy'); b = np.log10(b);
# Scatter plot echelle log : Chla Predict vs Chla True
plt.figure('Chla predict vs Chla true centré sur l''imagette')
plt.scatter(a, b, c='black')
plt.xlabel('Chla True',fontsize=14); plt.ylabel('Chla Predict', fontsize=14)
plt.xlim(-1.6, 0.1); plt.ylim(-1.25, 0.1)
plt.plot(range(-2,3),range(-2,3),'-r',linewidth=2.5)
#plt.title('Chla predict vs Chla true')
plt.savefig('../figures/examples/loss_evaluation/ytrue_vs_ypred_32x32center_log10.png')

# Histogramme de l'ecart entre chla Pred et chla True
c = np.subtract(b,a)
plt.figure('Histogramme de l''ecart entre chla predict et chla true')
plt.hist(c, bins=np.arange(-0.4,0.4,0.02), alpha=0.5)
plt.xlabel('Chla (Prédit - réel)',fontsize=14); plt.ylabel('Fréquence',fontsize=14)
#plt.title('Chla (Prédit - réel) centré sur l''imagette 64x64')
plt.savefig('../figures/examples/loss_evaluation/histo_ecart_log10_crop.png')

# calcul de rmse et coeff de correlation global
rmse_crop = rmse1(chlaTrue_crop, chlaPred_crop)
CorrCoef_crop = np.corrcoef(chlaTrue_crop, chlaPred_crop, bias=True)[0][1]
CorrCoef_crop
