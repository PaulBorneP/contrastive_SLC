# -*- coding: utf-8 -*-


import numpy as np
import random
from mvalab import *

#______________________________________________________________________________
#                           FONCTIONS SUPPORTS
#______________________________________________________________________________

def create_selection_area(h_ref,w_ref,h_patch,w_patch):
    '''
    Cette fonction fournit un masque booléen sous forme de grille de booléen
    dont la zone autorisée de sélection est définie par les valeurs 'True'.
    Cette zone constitue l'ensemble des indices dans lesquels on peut sélectionner
    l'origine d'un patch de taille (h_patch,w_patch) sans risque de débordement
    de la grille de référence de taille (h_ref,w_ref).
    L'origine d'un patch est définie comme le pixel du coin supérieur gauche
    du patch.

    Parameters
    ----------
    h_ref : integer
        Hauteur (height) de la grille de référence := nombre d'éléments selon l'axe 0
    w_ref : integer
        Largeur (width) de la grille de référence := nombre d'éléments selon l'axe 1.
    h_patch : integer
        Hauteur (height) du patch := nombre d'éléments selon l'axe 0.
    w_patch : integer
        Largeur (width) du patch := nombre d'éléments selon l'axe 1.

    Returns
    -------
    Area : array of boolean
        Tableau de valeurs booléennes. Une valeur 'True' indique que l'on point choisir
        l'indice en question comme position du pixel de référence du patch'
    ind_max_h : integer
        Valeur de l'INDICE MAXIMAL selon l'axe 0 auquel on peut positionner un patch.
    ind_max_w : integer
        Valeur de l'INDICE MAXIMAL selon l'axe 1 auquel on peut positionner un patch..
    '''
    
    ind_max_h = (h_ref-1) - (h_patch-2)
    ind_max_w = (w_ref-1) - (w_patch-2)
    Area                = np.full((h_ref, w_ref), True, dtype=bool)
    Area[ind_max_h:,:]  = False
    Area[:,ind_max_w:]  = False
    
    return Area, ind_max_h, ind_max_w


def choose_indice_ref_patch(ind_max_h, ind_max_w,method='random'):
    '''
    Cette fonction sélectionne la position du patch en fournissant la valeur des
    indices du pixel origine du patch. 
    Le pixel origine d'un patch est défini comme le pixel du coin supérieur gauche
    du patch.
    La position de ce pixel sera comprise dans le rectangle de dimension : 
        [0 --> ind_max_h, 0 --> ind_max_w]'
    

    Parameters
    ----------
    ind_max_h : integer
        Valeur de l'indice maximal de la région de sélection, selon l'axe 0 (height).
    ind_max_w : integer
        Valeur de l'indice maximal de la région de sélection, selon l'axe 1 (width).
    method : string, optional
        Identifiant de la méthode de séléction.
        The default is 'random'.

    Returns
    -------
    ind_h0_P : integer
        Valeur de l'indice du pixel de référence du patch selon l'axe 0 (height)
    ind_w0_P : integer
        Valeur de l'indice du pixel de référence du patch selon l'axe 1 (width).

    '''
    
    if method == 'random':
       ind_h0_P = random.randint(0, ind_max_h)
       ind_w0_P = random.randint(0, ind_max_w)
        
    
    return ind_h0_P, ind_w0_P


def create_forbiden_area(h_ref,w_ref,ind_h0_P,ind_w0_P,h0_P,w0_P):
    '''
    Cette fonction identifie une zone dans laquelle la sélection de nouveaux
    patchs est interdite. Cette zone est fournie sous la forme d'un masque 
    constitué d'un tableau de booléens. Une valeur 'False' dans ce masque indique 
    que la valeur des indices ne peut être attribuée au pixel origine d'un
    nouveau masque.
    Le pixel origine d'un patch est défini comme le pixel du coin supérieur gauche
    du patch.
    Le critère d'exclusion est le suivant : un nouveau patch ne peut partager
    aucun pixel avec le patch de référence.'
    
    
    Parameters
    ----------
    h_ref : integer
        Hauteur (height) de la grille := nombre d'éléments selon l'axe 0
    w_ref : integer
        Largeur (width) de la grille := nombre d'éléments selon l'axe 1.
    ind_h0_P : integer
        Valeur de l'indice du pixel origine du patch de référence selon l'axe 0 (height).
    ind_w0_P : integer
        Valeur de l'indice du pixel origine du patch de référence selon l'axe 1 (width).
    h0_P : integer
        Valeur de la hauteur du patch de référence selon l'axe 0 (height).
    w0_P : integer
        Valeur de la hauteur du patch de référence l'axe 1 (width).

    Returns
    -------
    Area : array of boolean
        Tableau de valeurs booléennes. Une valeur 'True' indique que l'on point choisir
        l'indice en question comme position du pixel de référence d'un nouveau patch

    '''

    Area = np.full((h_ref, w_ref), True, dtype=bool)
    Area[ind_h0_P : ind_h0_P + h0_P ,
         ind_w0_P : ind_w0_P + w0_P]  = False
    
    return Area

def extract_patch(img,ind_h0_P,ind_w0_P,h0_P,w0_P):
    '''
    Cette fonction extrait un patch d'une image 2D connaissant la position du
    pixel origine et la dimension du patch.
    Le pixel origine d'un patch est défini comme le pixel du coin supérieur gauche
    du patch.

    Parameters
    ----------
    img : np.array
        Image sous forme de tableau 2D
    ind_h0_P : integer
        Valeur de l'indice du pixel origine du patch de référence selon l'axe 0 (height).
    ind_w0_P : integer
        Valeur de l'indice du pixel origine du patch de référence selon l'axe 1 (width).
    h0_P : integer
        Valeur de la hauteur du patch de référence selon l'axe 0 (height).
    w0_P : integer
        Valeur de la hauteur du patch de référence l'axe 1 (width).

    Returns
    -------
    patch : np.array
        patch extrait de l'image.

    '''
    
    patch = img[ind_h0_P : ind_h0_P + h0_P,
                ind_w0_P : ind_w0_P + w0_P]
    
    
    return patch


#______________________________________________________________________________
#                           GENERATEUR DE PATCHS
#______________________________________________________________________________
class Sample_Generator:
    def __init__(self, data_path):
        
        data,w_img,h_img,nk_img,_ = imaread(data_path,ncan=0)
        self.path       = data_path
        self.teacher    = 'Real'
        self.student    = 'Imag'
        self.data       = data
        self.Re         = np.real(data)
        self.Im         = np.imag(data)
        self.img        = data
        self.h_img      = h_img
        self.w_img      = w_img
        
        del data,w_img,h_img,nk_img
        
    def create_sample(self,Channel,N_patch,h_patch,w_patch,method='random'):
        '''
        Cette fonction fournit un échantillon de patchs dont la définition est 
        cohérente de celle de l'équation (2) de l'article universitaire:
            - 1er position   : POSITIF | Teacher
            - 2°  position   : POSITIF | Student
            - 3° --> N_patch : NEGATIF | Student
        Le nombre de patch dans l'échantillon est "N_patch' issus de l'image du
        canal "Channel".
        La taille des patchs est (h_patch,w_patch).
        Le patch POSITIF est choisi selon la méthode "method".
        La dimension de l'échantillon est [N_patch x h_patch x w_patch]
            
        Parameters
        ----------
        Channel : integer
            Valeur du canal dans lequel est sélectionnée l'image dont on extrait 
            les patchs.
        N_patch : integer
            Nombre de patchs présents dans l'échantillon.
        h_patch : integer
            Nombre de pixels des patchs selon l'axe 0 (height)
        w_patch : integer
            Nombre de pixels des patchs selon l'axe 1 (width)
        method : string, optional
            Identifiant de la méthode utilisée pour extraire le patch POSITIF.
            The default is 'random'.

        Returns
        -------
        Sample : np.array [N_patch x h_patch x w_patch]
            Tableau contenant les patchs extraits. L'agencement des patchs est 
            cohérent de la convention utilisée dans l'équation (2) de l'article.
        Ind_Pori : Liste de tuples
            Liste contenant les indices des pixels origine de chaque patch sous forme de tuple.
            Les deux premiers tuples sont identiques car correspondant au patch POSITIF
            extrait sur Re(img) et Im(img).

        '''
        #----------------------------------------------------------------------
        # 0 - CREATION DES INDICES ORIGINES
        Ind_Pori = []
        
        # Patch POSITIF : Position du pixel origine 
        Area_1, ind_max_h, ind_max_w = create_selection_area(self.h_img,
                                                             self.w_img,
                                                             h_patch,
                                                             w_patch)
        ind_h0_P, ind_w0_P  = choose_indice_ref_patch(ind_max_h, ind_max_w, method)
        Area_2              = create_forbiden_area(self.h_img,
                                                   self.w_img,
                                                   ind_h0_P,ind_w0_P,
                                                   h_patch,w_patch)
        
        Ind_Pori +=[(ind_h0_P,ind_w0_P)]
        
        # Patchs NEGATIFS : Position des pixels origine
        Ind_Pori += [Ind_Pori[0]]   # la position du patch POSITIF est utilisée pour Re(img) et Im(img)
        for _ in range(N_patch-2):  # car deux positions sont déjà renseignées
            flag = False
            while flag==False:
                tmp_ind_h   = random.randint(0, ind_max_h)
                tmp_ind_w   = random.randint(0, ind_max_w)
                flag        = Area_1[tmp_ind_h,tmp_ind_w] & Area_2[tmp_ind_h,tmp_ind_w]
            Ind_Pori    +=[(tmp_ind_h,tmp_ind_w)]
        
        
        #----------------------------------------------------------------------
        # EXTRACTION DES PATCHS
        # La définition d'un échantillon est cohérente de l'équation (2) de l'article :
        #   - 1er position   : POSITIF | Teacher
        #   - 2°  position   : POSITIF | Student
        #   - 3° --> N_patch : NEGATIF | Student
        Sample = np.zeros((N_patch,h_patch,w_patch))
        
        # Extraction des patchs : Teacher
        if self.teacher == 'Real':
            img = self.Re[:,:,Channel]
        else:
            img = self.Im[:,:,Channel]
        
        ind_h = Ind_Pori[0][0]
        ind_w = Ind_Pori[0][1]
        Sample[0,:,:]   = extract_patch(img,
                                        ind_h,ind_w,
                                        h_patch,w_patch)
        
        # Extraction des patchs : Student
            # Le premier tuple de Ind_Pori identifie le patch Positif : 
            # il est utilisée deux fois (1x Teacher + 1x Student)
        if self.student == 'Real':
            img = self.Re[:,:,Channel]
        else:
            img = self.Im[:,:,Channel]
        for i in range(1,N_patch):
            ind_h = Ind_Pori[i][0]       
            ind_w = Ind_Pori[i][1]
            Sample[i,:,:]   = extract_patch(img,
                                            ind_h,ind_w,
                                            h_patch,w_patch)
                  
        return Sample, Ind_Pori
        
        

#______________________________________________________________________________
#                      ZONE DE TEST DES FONCTIONS
#______________________________________________________________________________
# ZONE DE TEST DES FONCTIONS
if False:
    # Test : Fonction validée
    Area, max_h, max_w =  create_selection_area(124,64,3,6)
    
    # Test : Fonction validée
    for _ in range (100):
        print(choose_indice_ref_patch(15, 8,method='random'))
        
        
    # Test : Fonction validée
    Area = create_forbiden_area(h_ref=109,
                                w_ref=37,
                                ind_h0_P=1,
                                ind_w0_P=5,
                                h0_P=2,
                                w0_P=8)
    
    
    # Test : Classe validée
    import os
    path     = os.path.join('../database/', 'PileDomancyTSX26.IMA')       
    GENERATEUR  = Sample_Generator(path)       

    Sampl, Ind = GENERATEUR.create_sample(Channel = 3,
                                      N_patch = 4,
                                      h_patch = 500,
                                      w_patch = 500,
                                      method='random')








