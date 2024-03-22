# -*- coding: utf-8 -*-


from IPython import get_ipython; 
get_ipython().run_line_magic('reset', '-sf')

# Définition du répertoire de travail
import os
os.chdir('c:\\[00]-DATA_CAK\\2023_MVA_ENS\\S2-Remote_Sensing\\PROJET\\Code')

import numpy as np
import random
import torch
from mvalab import *

import time

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

def Contrastive_Loss(batch_Teacher, batch_Student):
    """
    Cette fonction calcule la "contrastive loss" pour un des batchs de vecteurs
    issus du réseau "Professeur" et "Elève". Les batchs doivent respecter la convention
    suivante:
        - La taille du batch est identique pour les deux réseaux (B batchs).
        - Le vecteur de sortie est un vecteur 1D de même taille pour les deux réseaux (dim)
        - Le ième vecteur du batch venant du réseau "Professeur" est issu du traitement
        du même patch que le ieme vecteur du batch du réseau 'Etudiant' 
        (appliqué sur des données hétérogènes)
    Le calcul de la loss est fait en considérant succesivement chaque couple de vecteur
    i comme étant l'exemple positif et les B-1 autres comme les exemples négatifs.
    AINSI, IL EST NECESSAIER QUE CHAQUE PATCH SOIT ISSU D'UNE REGION DIFFERENTE
    DE L'MAGE.

    Parameters
    ----------
    batch_Teacher : tenseur pytorch
        Tenseur de dimension [B,dim] issu de l'inférence de B patchs différents 
        dans le réseau "Professeur".
        Avec le réseau choisi (ResNet34), on a dim = 100
    batch_Student : tenseur pytorch
        Tenseur de dimension [B,dim] issu de l'inférence de B patchs différents 
        dans le réseau "Elève".
        Avec le réseau choisi (ResNet34), on a dim = 100
    
    Returns
    -------
    loss : scalaire
        Moyenne sur les B batchs de la contrastive loss telle que définie dans
        l'équation (2) de l'article universitaire.
    """
    
    # Extraction des dimensions
    B_t, Dim_t   = np.array(batch_Teacher.shape)[0], np.array(batch_Teacher.shape)[1]
    B_s, Dim_s   = np.array(batch_Student.shape)[0], np.array(batch_Student.shape)[1]
    
    

    # Création d'un tenseur 3D pour vectoriser l'opération de soustraction paire à paire    
    T_rep = batch_Teacher.unsqueeze(1).expand(B_t, B_t, Dim_t)  # (B, 1, dim) -> (B, B, dim)
    S_rep = batch_Student.unsqueeze(0).expand(B_s, B_s, Dim_s)  # (1, B, dim) -> (B, B, dim)

    # Calcul de la norme euclidienne entre chaque paire de vecteurs
    normes = torch.norm(T_rep - S_rep, dim=2)
    numer = torch.diag(normes)
    denom = normes.sum(dim=1) - numer

    loss = -torch.log(numer/denom)
    loss = loss.mean()
    loss = loss.item()
    
    return loss

#______________________________________________________________________________
#                           GENERATEUR DE PATCHS : v1
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
        
        del data,w_img,h_img,nk_img
        
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
        en tant qu'attribut car sa création dure environ 2s et ne peut $donc être 
        réalisée à chaque création de patch.'
        '''
        Area, ind_list = create_cart_grid(self.img_size,patch_size,start)
        self.area_ref_M0 = Area
        del Area, ind_list
        
    def make_batch_M0(self,P, patch_size, channel_list='All'):
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
            La valeur par défaut ('All') permet de choisir tous les canaux
        

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
            
            
        return Batch, Area
            
    def make_batch_M1(self,P, patch_size, channel_list='All'):
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
            
            
        return Batch, Area
        
        



       




















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


# ZONE DE TEST DES CLASSES
if False:
    path    = os.path.join('../raw_databases/', 'PileDomancyTSX26.IMA')
    Batcher = BatchMaker(path)
    P           = 30
    patch_size  = (5,7)
    Batcher.init_area_ref(patch_size)
    
    
    # Test de la méthode M0 : validée
    Batcher.init_M0(patch_size,start='rand')
    batch,Area_batch = Batcher.make_batch_M0(P, patch_size, channel_list='All')
    batch,Area_batch = Batcher.make_batch_M0(P, patch_size, channel_list=[0,6,12])
    
    # Test de la méthode M1 : validée
    batch,Area_batch = Batcher.make_batch_M1(P, patch_size, channel_list='All')



if False:   # Script initial pour la création de la loss : important car pas évident à trouver
    # CETTE FACON DE FAIRE EST NICKEL POUR CALCULER LA LOSS SUR UN "BATCH" !!!
    # RESTE PLUS QU'A CALCULER LA SOMME ET FAIRE LE LOG
    
    # Exemple de dimensions
    N = 5
    dim = 1000
    
    # Création de deux tenseurs T1 et T2
    T1 = torch.randn(N, dim)
    T2 = torch.randn(N, dim)
    
    # Réplication des tenseurs pour pouvoir soustraire tous les paires de vecteurs en une seule opération
    T1_rep = T1.unsqueeze(1).expand(N, N, dim)  # (N, 1, dim) -> (N, N, dim)
    T2_rep = T2.unsqueeze(0).expand(N, N, dim)  # (1, N, dim) -> (N, N, dim)
    
    # Calcul de la norme euclidienne entre chaque paire de vecteurs
    normes = torch.norm(T1_rep - T2_rep, dim=2)
    numer = torch.diag(normes)
    denom = normes.sum(dim=1) - numer
    
    loss = -torch.log(numer/denom)
    loss = loss.mean()
    loss = loss.item()
    print(normes)
    print("Perte (loss) :", loss)
    
    tenseur = torch.randn(3, 4, 5)  # Un tenseur de dimensions 3x4x5
    
    
    
    # Obtenir les dimensions du tenseur
    dimensions = np.array(tenseur.shape)
    
    print("Dimensions du tenseur :", dimensions)



