import detectorlib
import matplotlib.pyplot as plt
import numpy as np

path="2021_All raw data_No_background.xlsx"

det=detectorlib.detector()
det.load_preprocess(path, 5)

#Per la rete neurale, acnhe l'ordine in cui inserisco le dimensioni risulta essere importante

det.reshape_tensor([63, 11], [17,  4, 0])


det.create_deep_model('conv3d')
det.fit_deep_model()


anomalies=det.detect_deep_anomalies()