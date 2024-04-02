import requests
import pandas as pd
import imageio
from tensorflow.keras.regularizers import l2
import moviepy.editor as mp
from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import base64
import requests
from requests.auth import HTTPBasicAuth
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
from keras.optimizers import Adam
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plot_keras_history
from sklearn.neighbors import KNeighborsClassifier as knn
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Reshape, LSTM, GRU, Conv1D, Conv3D, MaxPooling1D, MaxPooling3D
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import hyperopt
from datetime import timedelta
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


'''In this class we have to set the local variables to assign ath every index of our notations'''
class detector():

  def tuple_prod(self, tupla):
    prodotto = 1
    for dim in tupla:
        prodotto *= dim
    return prodotto
  

  '''Reads from excel file the data and append the sheets to the third index of the tensor: (temporal samples, features, sensors)'''
  def load_preprocess(self, path, sens_num):
        self.df = []
        self.xlsx_path=path
        for sheet_num in range(sens_num):  # Change to range(18) when you have all
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)
            # Dropping unnecessary columns
            
            sheet_df = sheet_df.drop(['Unnamed: 0', 'timestamp', 'sensor', 'off_ch1', 'off_ch2', 'off_ch3', 'off_ch4'], axis=1)
            self.df.append(sheet_df)
            
        
        self.df = np.array(self.df)
        self.df = np.nan_to_num(self.df)
        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        print(self.df)
        self.df = self.df.transpose(1, 0, 2)
     

  '''Given the desired index from the main, it reshape the df into a tensor as the user wants'''
  def reshape_tensor(self, temporal_indices, spatial_indices):
    if temporal_indices[0] == 0 and temporal_indices[1] == 0:
        print('Please set at least one temporal index different from 0')
        return
    # Verifica se almeno uno degli indici spaziali è diverso da zero
    if spatial_indices[0] == 0 and spatial_indices[1] == 0 and spatial_indices[2] == 0:
        print('Please set at least one spatial index different from 0')
        return
    # definisce nuovi inidici diversi da zero
    new_temporal_indices = [x for x in temporal_indices if x != 0]
    new_spatial_indices = [x for x in spatial_indices if x != 0]

    # Verifica se ci sono zeri e effettuare il reshape
    if 0 not in temporal_indices and 0 not in spatial_indices:
    # Se non ci sono zeri, esegui la reshape con tutti gli indici
      self.df = self.df.reshape(temporal_indices[0], temporal_indices[1], spatial_indices[0], spatial_indices[1], spatial_indices[2])
    else:
    # Se ci sono zeri, esegui la reshape con i nuovi indici senza gli zeri
      self.df = self.df.reshape(*new_temporal_indices, *new_spatial_indices)
      


  def create_model(self, string_model):
      try:
        if string_model == 'KMeans':
            self.str_model=string_model
            self.model = KMeans(n_clusters=3) # 7 parameters
        elif string_model == 'IsolationForest':
            self.str_model=string_model
            self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.2), random_state=42) # 8 parameters some bool
        elif string_model == 'SVM':
            self.str_model=string_model
            self.model = OneClassSVM(gamma='auto')  # 7 parameters
        elif string_model == 'LOF':
            self.str_model=string_model
            self.model = LocalOutlierFactor() # 8 parameters some useless
        elif string_model == 'linear':
            self.str_model=string_model
            self.model = LinearRegression() # 3 parameters only bool
        else:
            raise ValueError('Model name not recognized')
      except ValueError as e:
        print(f"Error creating model: {e}")
        return None


  def create_deep_model(self, string_model):
    try:
      if string_model == 'conv1d':
        self.str_model=string_model
        self.model = keras.Sequential([
          Conv1D(32, (3), activation='relu', padding='same', input_shape=(int(self.df.shape[1]), 1)),
          MaxPooling1D((2)),
          Flatten(),
          Dense(64, activation='relu'),
          Dense(1, activation='sigmoid'),
          Lambda(lambda x: x, output_shape=lambda s: s)  # Aggiungi questo layer Lambda
    ])
      elif string_model == 'conv2d':
        self.str_model=string_model
        self.model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.df.shape[1], self.df.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(self.df.shape[1] * self.df.shape[2], activation='sigmoid'),
        Reshape((self.df.shape[1], self.df.shape[2]))  # Modifica qui la dimensione dell'output
    ])
      elif string_model == 'conv3d':
        self.str_model=string_model
        self.model = keras.Sequential([
          Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(self.df.shape[1], self.df.shape[2], self.df.shape[3], 1)),
          MaxPooling3D((2, 2, 2)),
          Flatten(),
          Dense(64, activation='relu'),
    
          Dense(self.df.shape[1]*self.df.shape[2]*self.df.shape[3], activation='sigmoid'),
          Reshape((self.df.shape[1], self.df.shape[2], self.df.shape[3])),# Modifica qui la dimensione dell'output
          Lambda(lambda x: x, output_shape=lambda s: s)  # Aggiungi questo layer Lambda
    ])
        
      elif string_model == 'GRU1D':
        self.str_model=string_model
        self.model = keras.Sequential([
          GRU(64, input_shape=((self.df.shape[1]), 1), return_sequences=True),
          GRU(32),
          Dense((self.df.shape[1]), activation='linear'),# Modifica qui la dimensione dell'output
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'GRU2D':
        self.str_model=string_model
        self.model = keras.Sequential([
          GRU(64, input_shape=((self.df.shape[1:])), return_sequences=True),
          GRU(32),
          Dense(self.tuple_prod(self.df.shape[1:]), activation='linear'),# Modifica qui la dimensione dell'output
          Reshape((self.df.shape[1:])),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM1D':
        self.str_model=string_model
        self.model = keras.Sequential([
          LSTM(64, input_shape=((self.df.shape[1]), 1), return_sequences=True),
          LSTM(32),
          Dense((self.df.shape[1]), activation='linear'),# Modifica qui la dimensione dell'output
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM2D':
        self.str_model=string_model
        self.model = keras.Sequential([
          LSTM(64, input_shape=((self.df.shape[1:])), return_sequences=True),
          LSTM(32),
          Dense(self.tuple_prod(self.df.shape[1:]), activation='linear'),# Modifica qui la dimensione dell'output
          Reshape((self.df.shape[1:])),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
      else:
            raise ValueError('Model name not recognized')
      
    except ValueError as e:
          print(f"Error creating model: {e}")
          return None
    


  def fit_deep_model(self):
    self.model.compile(optimizer='adam', loss='mean_squared_error')
    history= self.model.fit(self.df, self.df, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
    plot_history(history)
    plt.show()

  def fit_model(self):
      self.model.fit(self.df)

  def fit_linear_model(self):
    self.model.fit(self.df, self.df)
 

  def detect_deep_anomalies_unsup(self):
    reconstructed = self.model.predict(self.df)
    mse = np.mean(np.power(self.df - reconstructed, 2), axis=1)
    rate=[]
    parameters = np.linspace(mse.min(), mse.max(), 1000)
    for i in range(len(parameters)):
      anomal = np.where(mse > parameters[i])
      if len(self.df.shape) == 2:
        rate.append((len(anomal[0]))/(self.df.shape[0]))
      elif len(self.df.shape) == 3:
        rate.append((len(anomal[0]))/(self.df.shape[0]*self.df.shape[2]))
      elif len(self.df.shape) == 4:
        rate.append((len(anomal[0]))/(self.df.shape[0]*self.df.shape[3]*self.df.shape[2]))        

    sns.set_style('darkgrid')
    plt.plot(parameters, rate)
    plt.scatter(parameters, rate)
    plt.title('Anomaly rate trend for combination: ' + str(self.str_model) + str(self.df.shape))
    plt.xlabel('mse threshold')
    plt.ylabel('anomaly rate ')
    plt.show()
    # calcolo derivata
    der = np.gradient(rate)
    ind_der = np.argmax(der)
    # Plot della derivata
    plt.plot(parameters, der)
    plt.xlabel('mse threshold')
    plt.ylabel('derivative')
    plt.show()
    print("Index max derivative:", ind_der)
    print("parameter in that point:", parameters[ind_der])
    anomalies_indices=[]
    anomalies_indices = np.where(mse > parameters[ind_der])
    print('Anomaly indices:', anomalies_indices)
    return(anomalies_indices)
  


  def fit_model(self):
    self.model.fit(self.df)

  def anomalies_sup(self):
    anomaly_percentage = 0.05  # 5%

    if self.str_model == 'SVM':
        self.fit_model()  # Training SVM model
        anomaly_scores = self.model.decision_function(self.df)
        num_anomalies = int(anomaly_percentage * len(self.df))
        anomaly_indices = np.where(anomaly_scores < 0)[0]  
        sorted_anomaly_indices = np.argsort(anomaly_scores[anomaly_indices])
        selected_anomaly_indices = anomaly_indices[sorted_anomaly_indices[:num_anomalies]]
        self.anomalies_indices = selected_anomaly_indices
        print(f"Anomalies indices for {self.model} {self.df.shape}:", self.anomalies_indices)
        print(f"Number of anomalies: {len(self.anomalies_indices)}")

    elif self.str_model == 'KMeans':
        self.fit_model()  # Training KMeans model
        distances = self.model.transform(self.df)
        mean_distance = np.mean(np.min(distances, axis=1))
        std_distance = np.std(np.min(distances, axis=1))
        threshold = mean_distance + 2 * std_distance  # Example: threshold as two std devs
        self.anomalies_indices = np.where(np.min(distances, axis=1) > threshold)[0]
        num_anomalies = int(anomaly_percentage * len(self.df))
        sorted_anomalies_indices = np.argsort(np.min(distances, axis=1))[::-1]
        self.anomalies_indices = sorted_anomalies_indices[:num_anomalies]
        print(f"Anomalies indices for {self.model} {self.df.shape}:", self.anomalies_indices)
        print(f"Number of anomalies: {len(self.anomalies_indices)}")

    elif self.str_model == 'LOF':
        self.fit_model()  # Training LOF model
        anomalies = self.model.fit_predict(self.df)
        self.anomalies_indices = np.where(anomalies < 0)[0]
        num_anomalies = int(anomaly_percentage * len(self.df))
        self.anomalies_indices = np.argsort(anomalies)[::-1][:num_anomalies]
        print(f"Anomalies indices for {self.model} {self.df.shape}:", self.anomalies_indices)
        print(f"Number of anomalies: {len(self.anomalies_indices)}")

    elif self.str_model == 'IsolationForest':
        self.fit_model()  # Training Isolation Forest model
        anomaly_scores = self.model.decision_function(self.df)
        num_anomalies = int(anomaly_percentage * len(self.df))
        anomaly_indices = np.where(anomaly_scores < 0)[0]  
        sorted_anomaly_indices = np.argsort(anomaly_scores[anomaly_indices])
        selected_anomaly_indices = anomaly_indices[sorted_anomaly_indices[:num_anomalies]]
        self.anomalies_indices = selected_anomaly_indices
        print(f"Anomalies indices for {self.model} {self.df.shape}:", self.anomalies_indices)
        print(f"Number of anomalies: {len(self.anomalies_indices)}")

    elif self.str_model == 'linear':
        self.fit_linear_model()  # Training linear regression model
        mse = np.mean(np.power(self.df - self.model.predict(self.df), 2), axis=1)
        percentile = np.percentile(np.abs(mse), (100*(1 - anomaly_percentage)))
        # Trova gli indici delle righe con residui che superano il 95% percentile
        self.anomalies_indices = np.where(np.abs(mse) > percentile)[0]
        print(f"Anomalies indices for {self.model} {self.df.shape}:", self.anomalies_indices)
        print(f"Number of anomalies: {len(self.anomalies_indices)}")


    else:
        print("Unknown model")



  def save_anomaly_indices(self):
    with open(f'anomalies {self.xlsx_path}/anomalies_' + str(self.model) + str(self.df.shape) + '.txt', 'w') as file:
        for indice in self.anomalies_indices:
            file.write(f"{indice}\n")



  def stamp_all_shape_anomalies(self, possible_shapes):
    for temporal_indices, spatial_indices in possible_shapes:
      self.reshape_tensor(temporal_indices, spatial_indices)
      self.anomalies_sup()
      self.save_anomaly_indices()


  def hyperopt_statistical_models(self, params):
    return



class sheet:
    def __init__(self):
        pass

    def load_timestamps(self, path, sens_num):
        self.df = []
        for sheet_num in range(sens_num):  # Change to range(18) when you have all
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)
            # Dropping unnecessary columns
            sheet_df = sheet_df['timestamp']
            self.df.append(sheet_df)
        return self.df
       

    def get_date(self, timestamp):
        return pd.to_datetime(timestamp, dayfirst=True).date()


    def find_discontinuity(self, *args):
        prev_date = None
        discs=[]
        dates=[]
        for idx, timestamps in enumerate(args):
            for i, timestamp in enumerate(tqdm(timestamps, desc=f'Analizing array {idx}', unit='timestamp')):
                curr_date = self.get_date(timestamp)

                if prev_date is None:
                    prev_date = curr_date
                    continue

                # Controlla se la data corrente è esattamente il giorno successivo alla data precedente
                if curr_date != prev_date + timedelta(days=1) and curr_date != prev_date + timedelta(days=0):
                    print(f"Discontinuity found on array {idx} on date : {prev_date}")
                    print(f"Delta days: {abs(curr_date - prev_date).days}")
                    discs.append(idx)
                    dates.append(prev_date)
                prev_date = curr_date

            # Resetta prev_date alla fine di ogni array per confrontare solo le date tra gli array
            prev_date = None
        return discs, dates


class printer():
  def __init__(self):
        pass
  
  def load(self, path, sens_num):
    self.df = []
    self.xlsx_path=path
    for sheet_num in range(sens_num):  # Change to range(18) when you have all
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)

            sheet_df = sheet_df.drop(['Unnamed: 0', 'off_ch1', 'off_ch2', 'off_ch3', 'off_ch4'], axis=1)
            self.df.append(sheet_df)


  def print_all(self):
    sns.set_style('darkgrid')
    cmap = plt.get_cmap('rainbow')
    normalize = Normalize(vmin=0, vmax=15)

    for i in tqdm(range(len(self.df)), desc="Elaborazione"):
        plt.figure(figsize=(15,10))

        for idx, col in enumerate(self.df[i].iloc[:, 2:].columns):
            color = cmap(normalize(idx))
            plt.plot(self.df[i]['timestamp'], self.df[i][col], label=col, color=color)

        plt.title(self.df[i]['sensor'].iloc[0])
        plt.xlabel('Time')
        plt.ylabel('Interactance')
        plt.legend()
        plt.savefig(f'graphs {self.xlsx_path}/sensor_{self.df[i]["sensor"].iloc[0]}.png')
        plt.close()


