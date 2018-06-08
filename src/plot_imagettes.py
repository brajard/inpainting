#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:22:48 2018

@author: arimoux
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
# outputname
outname = '../data/dataset_nn.nc'
name = 'model_4layers_4x4_morefilter'
exampledir = os.path.join('../figures/examples/',os.path.splitext(name)[0])

ds = xr.open_dataset(outname)

# parameters for figures
ind = 0
min_chloro = 0.05
SAVE = True
plot_full=True
plot_square=True
plot_hist = True
plot_hist_diff = True
plot_separated_figures = True
plot_colorbar = True
nanval = -1e5
nb_bins = 20

vmin = np.log10(min_chloro)
vmax = np.log10(ds.yt[ind,:,:].max())

isNotSquare = np.equal(ds.yt[ind,:,:],nanval)
yt0 = (1-isNotSquare)*ds.yt[ind,:,:]
yt0 = yt0.values
yt0 = yt0.squeeze()
yp0 = (1-isNotSquare)*ds.ypredict[ind,:,:]
yp0 = yp0.values
yp0 = yp0.squeeze()
xi,yi = np.nonzero(yt0)
yts = yt0[xi[0]:xi[-1]+1,yi[0]:yi[-1]+1]
yps = yp0[xi[0]:xi[-1]+1,yi[0]:yi[-1]+1]
yt_full = isNotSquare*ds.X[ind] + (1-isNotSquare)*ds.yt[ind]

#for histograms
yts_hist= np.concatenate(yts)
yps_hist= np.concatenate(yps)


if plot_full:
    fig, axes= plt.subplots(ncols=3)
    im0=axes[0].imshow(np.log10(ds.X[ind].squeeze()),vmin=vmin,vmax=vmax)
    axes[0].set_title("Input image",fontsize=14)
    im1=axes[1].imshow(np.log10(yt_full.squeeze()),vmin=vmin,vmax=vmax)
    axes[1].set_title("True Image",fontsize=14)
    im1=axes[2].imshow(np.log10(ds.yfinal[ind].squeeze()),vmin=vmin,vmax=vmax)
    axes[2].set_title("Inpainted Image",fontsize=14)
    filename = 'Image_' + str(int(ds.index[ind]))+'_full'
    title = 'Full images comparison'
    plt.suptitle(title,fontsize=16)
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))

if plot_square:
    fig, axes= plt.subplots(ncols=2)
    im0=axes[0].imshow(np.log10(yts.squeeze()),vmin=vmin,vmax=vmax)
    axes[0].set_title("True image",fontsize=14)
    im1=axes[1].imshow(np.log10(yps.squeeze()),vmin=vmin,vmax=vmax)
    axes[1].set_title("Inpainted Image",fontsize=14)
    filename = 'Image_' + str(int(ds.index[ind]))+'_square'
    title = 'Comparison of missing square'
    plt.suptitle(title,fontsize=16)
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))
        
if plot_hist:
    plt.figure(3)
    
    bins = np.linspace(np.minimum(yts.min(),yps.min()), np.maximum(yts.max(),yps.max()), nb_bins)
    plt.hist(yts_hist, bins, alpha=0.5, color='blue',label='True')
    plt.hist(yps_hist, bins, alpha=0.5, color='red',label='Inpainted')
    plt.legend(loc='upper right')
    plt.title("Histogram of Chlorophyll a concentration with "+ str(nb_bins) + ' bins')
    plt.xlim(np.minimum(yts.min(),yps.min()), np.maximum(yts.max(),yps.max()))
    plt.xlabel("Value")
    plt.ylabel("Instances")

    filename = 'Image_' + str(int(ds.index[ind]))+'_histo'
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))    

if plot_hist_diff:
    plt.figure(4)
    diff=yts_hist-yps_hist
    bins = np.linspace(diff.min(), diff.max(), nb_bins)
    plt.hist(diff, bins,alpha=0.5, color='red')
    plt.title("Error Histogram with "+ str(nb_bins) + ' bins')
    plt.xlim(diff.min(), diff.max())
    plt.xlabel("Value")
    plt.xlim(diff.min(), diff.max())
    plt.ylabel("Instances")

    filename = 'Image_' + str(int(ds.index[ind]))+'_histo_diff'
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))    

if plot_separated_figures:
    fig, ax = plt.subplots()    
    cax = ax.imshow(np.log10(ds.X[ind].squeeze()),vmin=vmin,vmax=vmax)
    ax.set_title("Input Image",fontsize=14)
    filename = 'Image_' + str(int(ds.index[ind]))+'_input_full'
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))
        
    fig, ax = plt.subplots()    
    cax = ax.imshow(np.log10(yt_full.squeeze()),vmin=vmin,vmax=vmax)
    ax.set_title("True Image",fontsize=14)
    filename = 'Image_' + str(int(ds.index[ind]))+'_true_full'
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))
        
    fig, ax = plt.subplots()    
    cax = ax.imshow(np.log10(ds.yfinal[ind].squeeze()),vmin=vmin,vmax=vmax)
    ax.set_title("Inpainted Image",fontsize=14)
    filename = 'Image_' + str(int(ds.index[ind]))+'_inpainted_full'
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))
        
    fig, ax = plt.subplots()    
    cax = ax.imshow(np.log10(yts.squeeze()),vmin=vmin,vmax=vmax)
    ax.set_title("True Image - Square",fontsize=14)
    filename = 'Image_' + str(int(ds.index[ind]))+'_true_square'
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))
    
    fig, ax = plt.subplots()    
    cax = ax.imshow(np.log10(yps.squeeze()),vmin=vmin,vmax=vmax)
    ax.set_title("Inpainted Image - Square",fontsize=14)  
    filename = 'Image_' + str(int(ds.index[ind]))+'_inpainted_square'
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))
     
if plot_colorbar:        
    fig, ax = plt.subplots()    
    cax = ax.imshow(np.log10(ds.X[ind].squeeze()),vmin=vmin,vmax=vmax)
    ax.set_title("Input Image",fontsize=14)
    log10_ticks=np.arange(vmin, np.round_(vmax+(vmax-vmin)/10,decimals=2),np.round_((vmax-vmin)/10,decimals=2))
    log10_ticks=np.insert(log10_ticks,0, -1000)
    lin_ticks = np.round_(np.power(10*np.ones(log10_ticks.shape),log10_ticks),decimals=2)
    cb=fig.colorbar(cax, ticks=log10_ticks, orientation='horizontal')
    cb.ax.set_xticklabels([str(lin_ticks[0]),str(lin_ticks[1]),
                           str(lin_ticks[2]),str(lin_ticks[3]),
                           str(lin_ticks[4]),str(lin_ticks[5]),
                           str(lin_ticks[6]),str(lin_ticks[7]),
                           str(lin_ticks[8]),str(lin_ticks[9]),
                           str(lin_ticks[10]),str(lin_ticks[11])])  # horizontal colorbar    
    cb.set_label("Chlorophyll a concentration [$mg/m^3$]",fontsize=14)
    filename = 'Image_' + str(int(ds.index[ind]))+'_colorbar'
    ax.set_visible(False)
    if SAVE:
        plt.savefig(os.path.join(exampledir,filename+'.png'))