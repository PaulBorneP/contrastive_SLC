# -*- coding: utf-8 -*-


import numpy as np

def calculate_PT_metrics(GT_array, INF_array):
    """
    Cette fonction calcule le nombre de faux positifs, de vrais positifs, de
    faux négatifs et de vrais négatifs.
    Ces métriques sont calculées sur la base de tableaux 2D emplis de 
        - 1 : les valeurs positives (changement identifié)
        - 0 : les valeurs négatives (pas de changement identifié)

    Parameters
    ----------
    GT_array : np.array
        Tableau contenant la vérité terrain
    INF_array : np.array
        Tableau contenant les inférences

    Returns
    -------
    TP : integer
        Nombre de "True Positive"
    FP : integer
        Nombre de "False Positive"
    TN : integer
        Nombre de "True Negative"
    FN : integer
        Nombre de "False Negative"
    """
    
    TP = np.sum(np.logical_and(GT_array == 1, INF_array == 1))
    FP = np.sum(np.logical_and(GT_array == 0, INF_array == 1))
    TN = np.sum(np.logical_and(GT_array == 0, INF_array == 0))
    FN = np.sum(np.logical_and(GT_array == 1, INF_array == 0))
    
    return TP, FP, TN, FN


def image_PT_metrics(GT_array, INF_array,
                     TP_col = [0, 255, 0],          # Couleur verte
                     FP_col = [255, 0, 0],          # Couleur rouge
                     TN_col = [0, 100, 0],          # Couleur verte foncée
                     FN_col = [255, 165, 0]):       # Couleur orange

    """
    Cette fonction identifie les pixels 'vrais positifs', 'faux positifs', etc.. 
    par une couleur spécifique et les replace dans une image RGB pour affichage.
    """
    
    TP_indices = np.where(np.logical_and(GT_array == 1, INF_array == 1))
    FP_indices = np.where(np.logical_and(GT_array == 0, INF_array == 1))
    TN_indices = np.where(np.logical_and(GT_array == 0, INF_array == 0))
    FN_indices = np.where(np.logical_and(GT_array == 1, INF_array == 0))
    
    image = np.zeros((GT_array.shape[0], GT_array.shape[1], 3), dtype=np.uint8)  # Initialisation de l'image RGB
    
    image[TP_indices[0], TP_indices[1], :] = TP_col
    image[FP_indices[0], FP_indices[1], :] = FP_col
    image[TN_indices[0], TN_indices[1], :] = TN_col
    image[FN_indices[0], FN_indices[1], :] = FN_col
    
    
    return image

