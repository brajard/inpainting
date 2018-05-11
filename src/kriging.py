 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:39:24 2018

@author: csoriot
"""
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import skgstat as skg


class Ordinary(BaseEstimator,RegressorMixin):
    """
    This class can extrapolate missing values and so complete an 2darray using the ordinary kriging method.
    By default, the number of point used in order to generate the semi-Variogram is 1000  wich seems to be good enought for a 64*64 image,
    and the theorical model is spherical.
    All the part on the Variogram is using the scikit-gstat library specially the Variogram.py program
    
    By default, the distance on wich values is used for kriged is 100 (so all the points will be used for a 64x64 image) but can be reduced in order to save time.
    this parameter will depend on the aspect of the semi-variogram
    
    If you have the real values behind the missing one, this calss can also return  :
        the r² coefficient 
        the RMSE
        the ration of variance 
    between the holed image and the complete masked image.
    """
    
    def __init__(self, distance = 100, nbPoints = 1000 , model = 'spherical'):  
        """
        param distance: float, distance for wich points have influence on the missing values
        param nbPoints: integer, number of points used to plot the experimantal semi-variogram
        param model: string, model used to plot the analytical semi-variogram
        """
        self.distance = distance
        self.nbPoints = nbPoints
        self.model = model 
        self.Im = None
        self.correlation = 0
        self.RMSE = 0
        self.sp = None
        self.cof = None
        self.V = None
        
    def fit(self, setImageHoled):
        """
        This function should average an analytical variogramm for several images wich will be use in the predict function.
        This part IS NOT finished
        
        param setImageHoled: 3darray, collection of several holed image
        """
    
        y = range(0,setImageHoled.shape[1])
        x = range(0,setImageHoled.shape[0])
        xv,yv = np.meshgrid(x,y)
 
        for index in range (len(setImageHoled)):
            lx = len(setImageHoled[0])
            ImageHoled = setImageHoled[index].reshape( (lx,lx) )
            complete = np.concatenate((xv.reshape( (-1,1) ), yv.reshape( (-1,1) ),
                                      ImageHoled.reshape( (-1,1) ) ), axis=1 )
            ok = np.isfinite(complete[:,2])
            carte = complete[ok,:]
            coordinates = carte[:,:2]
            values_no_nan = carte[:,2]           
            coef_plot = np.zeros([len(setImageHoled),2])
            nb_sample = np.random.choice(range(carte.shape[0]), self.nbPoints) 
            V = skg.Variogram(coordinates = coordinates[nb_sample], values = values_no_nan[nb_sample], normalize=False, model='spherical')
            coef_plot[:,0] += V.bins
            coef_plot[:,1] += V.experimental
        
        coef_plot = coef_plot/len(setImageHoled)
        
        self.cof = V.fit(coef_plot[:,0], coef_plot[:,1])[0]
        V.plot(show = False)
        return self.cof

    def _maskDistance(self, A, D):
        """
        Restrict the influcence of th
        """
        mask_distance = D < self.distance
        mask_distance = np.concatenate((mask_distance,np.array([True])),axis=0)
        A_distance= A[mask_distance,:]
        A_distance= A_distance[:,mask_distance]
        return mask_distance, A_distance
        
    def _build_A(self, carte):
        """
        Return the distance matrix A
        """
        A= np.zeros([len(carte)+1,len(carte)+1])
        A[:-1,:-1]=skg.distance.point_dist(carte[:,:2])
        la = A.shape[0]
        ca = A.shape[1]
        A = self.V.model(A.reshape(-1), self.V.cof[0], self.V.cof[1] )
        A = A.reshape((la,ca))
        A[:-1,len(A)-1]=1
        A[len(A)-1,:-1]=1
        A[len(A)-1,len(A)-1]=0
        del la, ca
        return A
    
    def _build_B(self, D_dist):
        """
        Return the vector B
        """
        B = np.ones([len(D_dist)+1])
        B[:-1] = self.V.model(D_dist, self.V.cof[0], self.V.cof[1])
        return B
   
    def predict(self, imageHoled, showVariogram = False, calculSp = False, imageMasked = None):
        """
        Calculates the missing values of the 2darray imageHoled using the ordinary kriging method
        
        
        param showVariogram: boolean, show the experimental and the analytical variogram if true
        param calculSp: boolean, calculates the standard deviation of each kriged point if true
        """
        
        y = range(0,imageHoled.shape[1])
        x = range(0,imageHoled.shape[0])
        xv,yv = np.meshgrid(x,y)
        ny = len(y)
        nx = len(x)
        full_image = np.concatenate((xv.reshape( (-1,1) ), yv.reshape( (-1,1) ),
                                  imageHoled.reshape( (-1,1) ) ), axis=1 )
        ok = np.isfinite(full_image[:,2])
        nok = np.isnan(full_image[:,2])
        carte = full_image[ok,:]
        coordinates = carte[:,:2]
        values_no_nan = carte[:,2]
        point_nan = full_image[nok,:2]
        nb_sample = np.random.choice(range(carte.shape[0]),self.nbPoints)
        
        #Construction of the theorical semivariogram using scikit-geostat
        self.V = skg.Variogram(coordinates = coordinates[nb_sample], values = values_no_nan[nb_sample], normalize = False, model = self.model)
        self.V.cof = self.V.fit(self.V.bins, self.V.experimental)[0]
        #self.V.plot(show = False, cof=self.V.cof)#Erreur sur show= showVariogram ? 
        A = self._build_A(carte)
        F = np.empty(len(point_nan))
       
        if calculSp:
            sp2 = np.zeros([len(point_nan)])
            carte_Sp = np.concatenate((xv.reshape( (-1,1) ), yv.reshape( (-1,1) ),
                                  imageMasked.reshape( (-1,1) ) ), axis=1 )
            
        for k in range(len(point_nan)): 
            D = np.sqrt(np.sum(np.square(point_nan[k]-carte[:,:2]),axis=1)) 
            D_dist = D[D<self.distance]
            B = self._build_B(D_dist)
            W = np.linalg.solve(self._maskDistance(A,D)[1] ,B)
            if calculSp: sp2[k] = sum(W*B)
            W = W/np.sum(W[:-1])
            F[k] = np.sum(W[:-1]*carte[self._maskDistance(A,D)[0][:-1],2]) 
            
        if calculSp:
            carte_Sp[nok,2] = sp2
            self.sp = np.reshape(carte_Sp[:,2],(nx,ny),order ='C')
            
        full_image[nok,2] = F            
        self.Im = np.reshape(full_image[:,2],(nx,ny),order='C')  
        return self.Im


    def r2(self, imageHoled, imageMasked):
        """
        Returning the coerrlation coefficient r² between the 2darays imageHoled and imageMasked
        The missing values in imageMasked are nan val
        """
        try:
            fi = self.Im[np.isfinite(imageMasked)]
        except ValueError:
            self.predict(imageHoled)
            fi = self.Im[np.isfinite(imageMasked)]            
        yi = imageMasked[np.isfinite(imageMasked)]
        moy = np.mean(yi)
        ssres = np.sum(np.square(yi-fi))
        sstot = np.sum(np.square(yi-moy))
        self.correlation = 1-(ssres/sstot)                
        return self.correlation        
        
    def rapportVariance(self, imageHoled, imageMasked):
        """
        Return the ratio of variance between the kriged image and the real image
        """
        try :
            varP = np.var(self.Im[np.isfinite(imageMasked)])      
        except ValueError:
            self.Im = self.predict(imageHoled)
            varP = np.var(self.Im[np.isfinite(imageMasked)])           
        varT = np.var(imageMasked[np.isfinite(imageMasked)])
        return (varT/varP)
    
    def rmse(self, imageHoled, imageMasked):
        """
        Return the root mean square ecart between the kriged image and the real image
        """
        try:
            fi = self.Im[np.isfinite(imageMasked)]
        except ValueError:
            self.predict(imageHoled)
            fi = self.Im[np.isfinite(imageMasked)]           
        yi = imageMasked[np.isfinite(imageMasked)]
        self.RMSE = np.sqrt(np.sum(np.square(fi-yi))/len(yi))
        return self.RMSE
    
    def calulSp(self, imageHoled, imageMasked):
        """
        Return the image with the standart deviation on each missing values
        """
        self.predict(imageHoled, calulSp=True, imageMasked=imageMasked)
        return self.sp

    def vario_r2(self):
        """
        #Return the Pearson correlation of the theoritical Variogram
        """
        return self.V.r
    
    def vario_rmse(self):
        """
        #Return the RMSE of the theoritical Variogram
        """
        return self.V.RMSE


