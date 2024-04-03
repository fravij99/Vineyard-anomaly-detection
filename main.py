import detectorlib
import matplotlib.pyplot as plt
import numpy as np

path="2022_LOU_AZE_field_maturation_final_matrix.xlsx"
path2='2021_All raw data_No_background.xlsx'

possible_shapes2=[
([63, 0], [16*11*6, 0, 0]), 
([63*11, 0], [16*6, 0, 0]),	
([63*11*6, 0], [16, 0, 0]),	
([63*11*16, 0], [6, 0, 0]),	
([16*6, 0], [63*11, 0, 0]),	
([6, 0], [63*11*16, 0, 0]),	
([16, 0], [63*11*6, 0, 0]),	
]

possible_shapes=[
([41, 0], [16*11*10, 0, 0]), 
([41*11, 0], [16*10, 0, 0]),	
([41*11*10, 0], [16, 0, 0]),	
([41*11*16, 0], [10, 0, 0]),	
([16*10, 0], [41*11, 0, 0]),	
([10, 0], [41*11*16, 0, 0]),	
([16, 0], [41*10*11, 0, 0]),	
]

possible_models={'linear', 'KMeans', 'SVM', 'LOF', 'IsolationForest'}  #'linear', 'KMeans', 'SVM', 'LOF', 'IsolationForest' 
det=detectorlib.detector()
det.load_preprocess(path2, 6)

for model in possible_models:
    det.create_model(model)
    #Per la rete neurale, acnhe l'ordine in cui inserisco le dimensioni risulta essere importante
    det.stamp_all_shape_anomalies(possible_shapes2)