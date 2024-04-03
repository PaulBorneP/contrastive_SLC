# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
from database.utils import imaread
from mvalab import *
from transforms import *


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

def create_cart_grid(img_size,patch_size,start='ref'):
    '''
    Cette fonction crée une grille cartesienne des position du pixel de référence
    permettant d'obtenir un agencement cartesien des patchs: 
        les patchs seront alignés côte à côte, sans chevauchement
        

    Parameters
    ----------
    img_size : tuple (h_img, w_img)
        Taille de l'image : h := nombre de pixel du patch selon l'axe 0 (hauteur)
                          : w := nombre de pixel du patch selon l'axe 1 (largeur)
    patch_size : tuple (h_patch, w_patch)
        Taille du patch : h := nombre de pixel du patch selon l'axe 0 (hauteur)
                        : w := nombre de pixel du patch selon l'axe 1 (largeur)
    start : string
        Méthode de sélection du point de référence de la grille:
            'ref'  := la grille commence à l'indice (0,0)
            'rand' := la grille commence à une position aléatoire dans la zone
                      [0:h_patch , 0:w_patch]

    Returns
    -------
    Area : tableau de booléen de dimension 'img_size'
        Tableau identifiant les positions autorisées (indices) des pixels de référence
        avec la valeur 'True'
    ind_list : liste de tuple
        Liste fournissant les positions autorisées (indices) des pixels de référence.

    '''
    # 1 - Choix de la position de référence : (ind_h , ind_w)
    if start == 'ref':
        ref_pos = (0,0)
    if start == 'rand':
        ref_pos = (random.randint(0, patch_size[0] - 1),
                   random.randint(0, patch_size[1] - 1))
        
    # 2 - Création de la liste des position des pixels de référence :
    #     les patchs seront ordonnés selon une grille cartesienne sans chevauchement
    h = np.arange(ref_pos[0], img_size[0], patch_size[0])
    w = np.arange(ref_pos[1], img_size[1], patch_size[1])
    
    H, W        = np.meshgrid(h, w)
    indices     = np.column_stack((H.flatten(), W.flatten()))
    ind_list    = [tuple(row) for row in indices]
    del H,W,indices
    
    # 3 - Création du tableau de vérité
    # Créer un tableau de booléens rempli de False de taille (20, 30)
    Area = np.full(img_size, False, dtype=bool)
    for ind in ind_list:
        Area[ind] = True
       
    return Area, ind_list

#______________________________________________________________________________
#                           GENERATEUR DE PATCHS : v2
#______________________________________________________________________________
class BatchMaker:
    def __init__(self,data_path):
        data,w_img,h_img,nk_img,_ = imaread(data_path,ncan=0)
        self.path       = data_path
        self.data       = data
        self.Re         = np.real(data)
        self.Im         = np.imag(data)
        self.h_img      = h_img
        self.w_img      = w_img
        self.img_size   = (self.h_img,self.w_img)
        self.channel    = nk_img
        
        self.area_ref    = 'None'
        self.area_ref_M0 = 'None'
        self.img_het     = []
        
        del data,w_img,h_img,nk_img
    
    def sym_ReIm(self):
        """
        Cette méthode appelle la fonction "symetrise_real_and_imaginary_parts"
        du fichier "transforms.py". Elle ne s'utilise que sur des les parties
        réelles et imaginaires d'une même image. Cette méthode doit donc être appelée
        juste après la création du batch. Elle est irréversible et modifie les 
        attributs" "self.Re" et "self.Im".
        """
        sym_Re, sym_Im =\
            symetrise_real_and_imaginary_parts(self.Re, self.Im)
        self.Re = sym_Re 
        self.Im = sym_Im
        
    def init_area_ref(self,patch_size):
        '''
        Cette crée la grille de vérité de référence, dépendant de la taille de 
        l'imag et de la taille du patch'

        Parameters
        ----------
        patch_size : tuple (h,w)
            Taille du patch : h := nombre de pixel du patch selon l'axe 0 (hauteur)
                            : w := nombre de pixel du patch selon l'axe 1 (largeur)
        '''
        Area_REF, _, _ = create_selection_area(self.h_img,self.w_img,patch_size[0],patch_size[1])
        self.area_ref = Area_REF
        del Area_REF
    
    def init_M0(self,patch_size,start='ref'):
        '''
        Cette méthode crée la grille de référence pour la méthode n°0 et la stocke
        en tant qu'attribut car sa création dure environ 2s et ne peut donc être 
        réalisée à chaque création de patch.'
        '''
        Area, ind_list = create_cart_grid(self.img_size,patch_size,start)
        self.area_ref_M0 = Area
        del Area, ind_list
        
    def make_batch_M0(self,P, patch_size, 
                      channel_list='All', preproc_norm = True):
        '''
        Cette fonction crée "P" patchs de taille "patch_size" selon la méthode suivante:
            - les patchs sont choisis aléatoirement selon une grille cartesienne sans recouvrement
            - la position des patchs est choisie aléatoirement dans cette grille, sans redondance
            - à chaque position choisie, le patch est sélectionné aléatoirment  
              selon l'un des canaux de la liste "channel_list"

        Parameters
        ----------
        P : entier
            Nombre de patch a créer
        patch_size : tuple (h,w)
            Taille du patch : h := nombre de pixel du patch selon l'axe 0 (hauteur)
                            : w := nombre de pixel du patch selon l'axe 1 (largeur)
        channel_list : liste d'entiers
            Les patchs sont choisis dans les images de canal appartenant à cette liste.
            La valeur par défaut ('All') permet de choisir tous les canaux.
        preproc_norm : booléen
            Réalisation du pré-processing MERLIN sur la normalisation des patchs.
            Par défaut ce pré-traitement est réalisé
       

        Returns
        -------
        Batch : np.array (P,2,patch_size[0],patch_size[1])
            Batch de P couples d'images {Re,Im} représentant des zones toues différentes
            entre-elles.
        Area : np.array (self.h_img,self.w_img) de booléns
            Tablea de booléens indiquant les positions des pixesl origine encore
            disponibles pour extraction de patchs

        '''
        
        # 1 - Créer la grille de vérité : ajout de la grille cartésienne et de référence
        Area    = self.area_ref & self.area_ref_M0
        
        # 2 - Extraction des patchs : indices des pixels origines disponibles
        ind_true = np.where(Area)
        ind_list = list(zip(ind_true[0], ind_true[1]))
        
        
        # 2 - Extraction des patchs : dimensions du batch
        if len(ind_list) < P:
            print(f"Le nombre de patchs fournis ({len(ind_list)}) sera inférieur à celui demandé ({P})")
            P = len(ind_list)
        Batch = np.zeros((P,2,patch_size[0],patch_size[1]))
        
        # 2 - Extraction des patchs : canaux disponibles
        if channel_list=='All':
            channel_list = list(range(self.channel))
        
        # 2 - Extraction des patchs : remplissage du BATCH
        for b in range(P):
            c = random.randint(0, len(channel_list)-1)
            c = channel_list[c]
            i= random.randint(0, len(ind_list)-1)
                        
            Batch[b,0,:,:] = extract_patch(self.Re[:,:,c],
                                  ind_list[i][0],ind_list[i][1],
                                  patch_size[0],patch_size[1])
            Batch[b,1,:,:] = extract_patch(self.Im[:,:,c],
                                  ind_list[i][0],ind_list[i][1],
                                  patch_size[0],patch_size[1])
            
            Area[ind_list[i]] = False   # la position choisie n'est plus disponible
            _ = ind_list.pop(i)         # idem
            
        # - Pré-traitements
        if preproc_norm:
            from transforms import sar_normalization
            Batch = sar_normalization(Batch)
            
        return Batch, Area
            
    def make_batch_M1(self,P, patch_size, 
                      channel_list='All', preproc_norm=True):
        '''
        Cette fonction crée "P" patchs de taille "patch_size" selon la méthode suivante:
            - les patchs sont choisis aléatoirement dans toute l'image
            - les patchs ne peuvent pas être extraits dans une zone identique mais
            des recouvrements partiels sont possibles.

        Parameters
        ----------
        P : entier
            Nombre de patch a créer
        patch_size : tuple (h,w)
            Taille du patch : h := nombre de pixel du patch selon l'axe 0 (hauteur)
                            : w := nombre de pixel du patch selon l'axe 1 (largeur)
        channel_list : liste d'entiers
            Les patchs sont choisis dans les images de canal appartenant à cette liste.
            La valeur par défaut ('All') permet de choisir tous les canaux        
        preproc_norm : booléen
            Réalisation du pré-processing MERLIN sur la normalisation des patchs.
            Par défaut ce pré-traitement est réalisé
        
        Returns
        -------
        Batch : np.array (P,2,patch_size[0],patch_size[1])
            Batch de P couples d'images {Re,Im} représentant des zones toues différentes
            entre-elles.
        Area : np.array (self.h_img,self.w_img) de booléns
            Tablea de booléens indiquant les positions des pixesl origine encore
            disponibles pour extraction de patchs
        '''
        
        # 1 - Extraction des indices disponibles pour les pixels origines
        Area     = self.area_ref
        ind_true = np.where(Area)
        ind_list = list(zip(ind_true[0], ind_true[1]))
        
        # 2 - Extraction des patchs : dimensions du batch
        if len(ind_list) < P:
            print(f"Le nombre de patchs fournis ({len(ind_list)}) sera inférieur à celui demandé ({P})")
            P = len(ind_list)
        Batch = np.zeros((P,2,patch_size[0],patch_size[1]))
        
        
        # 2 - Extraction des patchs : canaux disponibles
        if channel_list=='All':
            channel_list = list(range(self.channel))
        
        # 2 - Extraction des patchs : remplissage du BATCH
        for b in range(P):
            c = random.randint(0, len(channel_list))
            c = channel_list[c]
            i= random.randint(0, len(ind_list))
                        
            Batch[b,0,:,:] = extract_patch(self.Re[:,:,c],
                                  ind_list[i][0],ind_list[i][1],
                                  patch_size[0],patch_size[1])
            Batch[b,1,:,:] = extract_patch(self.Im[:,:,c],
                                  ind_list[i][0],ind_list[i][1],
                                  patch_size[0],patch_size[1])
            
            Area[ind_list[i]] = False   # la position choisie n'est plus disponible
            _ = ind_list.pop(i)         # idem
            
        # - Pré-traitements
        if preproc_norm:
            from transforms import sar_normalization
            Batch = sar_normalization(Batch)
           
        return Batch, Area
    
    def make_img_het(self,channel):
        """
        Cette fonction crée une pile de deux images hétérogènes à partir de
        la pile des image de partie Réelle et Imaginaire. 
            - La première image est une image de partie réelle choisie 
            comme celle du canal 'channel[0]'.
            - La seconde image est une image de partie imaginaire choisie 
            comme celle du canal 'channel[&]'.

        Parameters
        ----------
        channel : tuple
            Indice des canaux desquels sont exatraites les images de partie 
            Réelle et Imaginaire.
        """

        self.img_het = np.stack((self.Re[:,:,channel[0]], self.Re[:,:,channel[1]]))
        
    def make_batch_inference(self,P, patch_size,ind_avai,
                             preproc_norm = True):
        """
        Cette fonction crée un batch de patchs sous le format [P,2,h_patch,w_patch].
        La position des pixels de référence de chaque patch est faite selon la méthode
        suivante : les 'P' premiers tuples de la liste de tuple 'ind_avai' fournissent
        la position des pixels origine.
        Les patchs sont extraits de l'attribut ''self.img_het' (cf. méthode 'make_img_het').

        Parameters
        ----------
        P : integer
            Nombre de patch a créer
        patch_size : tuple (h,w)
            Taille du patch : h := nombre de pixel du patch selon l'axe 0 (hauteur)
                            : w := nombre de pixel du patch selon l'axe 1 (largeur)
        ind_avai : list of tuple
            Liste des positions des pixels de référence encore disponibles.
            Chaque position est fournie selon le formalisme (h_pixel,w_pixel).
            
        preproc_norm : booléen
            Réalisation du pré-processing MERLIN sur la normalisation des patchs.
            Par défaut ce pré-traitement est réalisé

        Returns
        -------
        Batch : np.array [P,2,h_patch,w_patch]
            Tableau fournissant les P patchs extraits de dimension (h_patch,w_patch).
            [:,0,:,:] : parties Réelles
            [:,1,:,:] : parties Imagainaires     
        ind_Batch : list of tuple
            Liste de taille 'P' fournissant les positions des pixels origines
            de chaque patch paire de patch
        ind_remaining : list of tuple
            Liste de taille fournissant les positions des pixels origines
            encore disponibles après création des 'P' patchs.
        """
        # 1 - Gestion des listes d'indices
        ind_Batch       = list(ind_avai[:P])
        ind_remaining   = ind_avai.copy()
        del ind_remaining[:10]
        
        # 2 - Création des patchs : [P,2,h_patch,w_patch]
        Batch = []
        for pos in ind_Batch:
            patch = self.img_het[:,pos[0]:pos[0]+patch_size[0],
                                 pos[1]:pos[1]+patch_size[1]]
            Batch.append(patch)
        Batch = np.array(Batch)
        
        # - Pré-traitements
        if preproc_norm :
            from transforms import sar_normalization
            Batch = sar_normalization(Batch)
        
        return Batch, ind_Batch, ind_remaining
    
    



