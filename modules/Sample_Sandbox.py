# -*- coding: utf-8 -*-

from IPython import get_ipython; 
get_ipython().run_line_magic('reset', '-sf')

# Définition du répertoire de travail
import os
import numpy as np
import random
from mvalab import *


if False : # EXPLORATION MANUELLE DES DONNEES
    imgName = os.path.join('../database/', 'PileDomancyTSX26.IMA')
    im_0,w_0,h_0,nk_0,nkt_0 = imaread(imgName,ncan=0)         # Lit tous les canaux et renvoie un tableau [r*c*nb_canaux]
    im_5,w_5,h_5,nk_5,nkt_5 = imaread(imgName,ncan=5)         # Lit que le canal 5
    
    
    # La fonction imaread renvoie 5 sorties:
        # img : les valeurs numériques contenues dans les images 
        #     : taleau (w,h,nk) si nk>1
        # w   : la valeur des éléments selon l'axe 0 du tableau (w:=width:=largeur)
        #     : un scalaire
        # h   : la valeur des éléments selon l'axe 1 du tableau (h:=height:=hauteur)
        #     : un scalaire
        # nk  : le nombre de canaux de l'image'
        #     : un scalaire
        # nktemps  : je ne sais pas ce que c'est'
        #           : un scalaire
    
      
    if False:   # Visualisation avec les fonctions de mvalab
        k = 3  # la valeur du seuil sur l'écart-type (pour borner la dynamique de l'image)
        for i in range(np.shape(im_0)[2]):
            visusar(np.abs(im_0[:,:,i]),k)
            visusar(np.real(im_0[:,:,i]),k)
            visusar(np.imag(im_0[:,:,i]),k)
            visusar(np.angle(im_0[:,:,i]),k)
         
        # Visualisation avec matplotlib
        # Pour changer la taille par défaut de toutes les figures à [9pouces,9pouces]
        plt.rcParams['figure.figsize'] = [9, 9]      
        plt.imshow(np.angle(im_0[:,:,i]))



