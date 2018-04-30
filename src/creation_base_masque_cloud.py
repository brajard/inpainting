# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:56:55 2018

pour faire une nouvelle base de masques en jouant sur le pourcentage des valeurs manquantes 
et l'eloignement sur les bords 
Placement Random des Masques.

err_cloud
err_invalide
err_land
dx
@author: DELL
"""
import numpy as np
from skimage import morphology
import xarray as xr
from numpy import genfromtxt
from sklearn.preprocessing import Imputer

imp = Imputer(strategy="mean")
ds = xr.open_dataset('../data/medchl.nc')
data = genfromtxt('../data/medchl.csv', delimiter=',')

dx=dy=5
err_cloud=1
err_invalide=1
err_land=1
surface_min=100 # surface min de pixels nan (nuage) accepté 
surface_max=500 # surface min de pixels nan (nuage) accepté 
nombre_images = 10000
a=[]
# Selection des imagettes chla selon leurs de chaque type de données spécifié ci-dessus
for i in range(len(data[:,1])-1):
    if ((data[i+1,3]<=err_cloud)  & (data[i+1,4]<=err_invalide)& (data[i+1,5]<=err_land)):
        a.append(i)
keep = np.zeros(shape=ds.dims['index']).astype(bool)
keep[[a]] = True
chlanew = ds.chla[keep]
flagnew= ds.flags[keep]

ds_new = xr.Dataset({'chla':chlanew,'flags':flagnew})
xx=np.zeros((1, 64, 64),dtype=bool)

k=0
while (k<nombre_images):
    i = np.random.randint(0,np.shape(ds_new.index.values)[0])
	# pour chaque image 
    x=np.zeros(np.shape(ds_new.chla[i,:,:]),dtype=bool) # initialisation par une matrice de False
    y=np.argwhere(np.isnan(ds_new.chla.values[i,:,:]))  # coords des nan ie des nuages dans l'image
    x[y[:,0],y[:,1]]=True  # on remplace les nan ie les nuages par des "True"
    m=morphology.label(x, connectivity=2)  # Selection des regions (nuage) a contours fermé et les remplir par un chiffre donné 
    print(i,k)
    for j in range(np.max(m)): # mex pour parcourir toutes les régions (nuages)
	# pour chaque region fermé (nuage entier)
        c=np.where(m==j) # Selectionne les coords 
		# condition de selection du nuage sur la proximité p/r aux bords et la taille du nuage
        if ( ((np.shape(c[0])[0]) > surface_min) and ((np.shape(c[0])[0]) < surface_max) and (np.min(c)>dx) and (np.max(c) <63-dx)):
            aa=np.zeros((1, 64, 64),dtype=bool) # initialisation du mask
            aa[0,c[0],c[1]]=True                # Mise en place du mask
            xx=np.append(xx,aa,axis=0)          # pourquoi pas "extend"?
    k=k+1
data_mask=xr.DataArray(xx[1:]) # Sauvegarde dans le data array
data_mask.to_netcdf('../data/data_mask.nc') #Sauvegarde dans le dataset des mask