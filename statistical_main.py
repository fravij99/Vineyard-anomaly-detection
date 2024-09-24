import detectorlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

path="2022_LOU_AZE_field_maturation_final_matrix.xlsx"
path2='2021_All raw data_No_background.xlsx'
path3="2020_TN_Field_alldata_SD_divisi.xlsx"

'''MOLTO IMPORTANTE: L'ORDINE CON CUI SIS SCRIVONO GLI INDICI NON SI DEVE CAMBIARE ASSOLUTAMENTE
[([63], [16, 11, 6]), 
([63, 11], [16, 6]),	
([63, 11, 6], [16]),	
([63, 16, 11], [6]),	
([16, 6], [63, 11]),	
([6], [63, 11, 16]),	
([16], [6, 11, 63])]
'''

possible_shapes3=[
([42], [16, 11, 12]), 
([42, 11], [16, 12]),	
([42, 11, 12], [16]),	
([42, 11, 16], [12]),	
([16, 12], [42, 11]),	
([12], [42, 11, 16]),	
([16], [12, 11, 42]),	
]

possible_shapes2=[
([63], [16, 11, 6]), 
([63, 11], [16, 6]),	
([63, 11, 6], [16]),	
([63, 11, 16], [6]),	
([16, 6], [63, 11]),	
([6], [63, 11, 16]),	
([16], [6, 11, 63]),	
]

possible_shapes=[
([41], [16, 11, 10]),   # anomalia climatica
([41, 11], [16, 10]),   # anomalia atmosferica
([41, 11, 10], [16]),   # anomalia di singolo sensore in un determinato timepoint
([41, 11, 16], [10]),	# anomalia di singolo canale in un determinato timepoint
([16, 10], [41, 11]),   # anomalia di singolo canale continuata nel tempo
([10], [41, 11, 16]),	# anomalia di singolo sensore continuata nel tempo
([16], [10, 11, 41]),
 	] 	# anomalia di singolo canale per tutti i sensorimi

det = detectorlib()
det.load_preprocess(path3, 12)
possible_models={'PCA, SVM, IsolationForest, LOF, KMeans'}


for model in tqdm(possible_models, desc="Creating models"):
    det.create_statistical_model(model)
    # Per la rete neurale, anche l'ordine in cui inserisco le dimensioni risulta essere importante

    det.stamp_all_shape_anomalies_PCA(possible_shapes3)



