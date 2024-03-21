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
from ruptures.metrics import hausdorff
from ruptures.costs import cost_normal, cost_constant
from ruptures.detect import bocd
from ruptures.utils import pairwise



'''In this class we have to set the local variables to assign ath every index of our notations'''
class detector():


  def tuple_prod(self, tupla):
    prodotto = 1
    for dim in tupla:
        prodotto *= dim
    return prodotto
  
  '''Reads from excel file the data and append the sheets to the third index of the tensor: (temporal samples, features, sensors)'''
  def load_preprocess(self, path, sens_num):
    self.df=[]
    for sheet_num in range(1, sens_num):  # 18 fogli numerati da 1 a 18
      sheet_df = pd.read_excel(path, sheet_name=sheet_num)
      self.df.append(sheet_df)
      self.df[sheet_num-1]=self.df[sheet_num-1].drop(['timestamp','sensor', 'off_ch1' , 'off_ch2', 'off_ch3' , 'off_ch4'], axis=1)
    self.df=np.array(self.df)
    self.df=(self.df-self.df.min())/(self.df.max()-self.df.min())
    self.df=self.df.transpose(1, 0, 2)

  
  '''Given the desired index from the main, it reshape the df into a tensor as the user wants'''
  def reshape_tensor(self, temporal_indices, spatial_indices):
    if temporal_indices[0] == 0 and temporal_indices[1] == 0:
        print('Please set at least one temporal index different from 0')
        return

    # Verifica se almeno uno degli indici spaziali è diverso da zero
    if spatial_indices[0] == 0 and spatial_indices[1] == 0 and spatial_indices[2] == 0:
        print('Please set at least one spatial index different from 0')
        return
    # define new indices without zeros
    new_temporal_indices = [x for x in temporal_indices if x != 0]
    new_spatial_indices = [x for x in spatial_indices if x != 0]

    # Verificare se ci sono zeri e effettuare la reshape
    if 0 not in temporal_indices and 0 not in spatial_indices:
    # Se non ci sono zeri, esegui la reshape con tutti gli indici
      self.df = self.df.reshape(temporal_indices[0], temporal_indices[1], spatial_indices[0], spatial_indices[1], spatial_indices[2])
    else:
    # Se ci sono zeri, esegui la reshape con i nuovi indici senza gli zeri
      self.df = self.df.reshape(*new_temporal_indices, *new_spatial_indices)
      


  def create_model(self, string_model):
      try:
        if string_model == 'KMeans':
            self.model = KMeans(n_clusters=5)
        elif string_model == 'IsolationForest':
            self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.2), random_state=42)
        elif string_model == 'SVM':
            self.model = OneClassSVM(gamma='auto')
        elif string_model == 'LOF':
            self.model = LocalOutlierFactor()
        elif self.model == 'bayesian':
            self.model = bocd(model='normal')
        else:
            raise ValueError('Model name not recognized')
      except ValueError as e:
        print(f"Error creating model: {e}")
        return None

  def fit_deep_model(self):
    self.model.compile(optimizer='adam', loss='mean_squared_error')
    history= self.model.fit(self.df, self.df, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    plot_history(history)
    plt.show()

  def fit_model(self):
      self.model.fit(self.df)

  def fit_ridge(self):
    self.model.fit(self.df, self.df)


  def create_deep_model(self, string_model):
    try:
      if string_model == 'conv1d':
        self.model = keras.Sequential([
          Conv1D(32, (3), activation='relu', padding='same', input_shape=(int(self.df.shape[1]), 1)),
          MaxPooling1D((2)),
          Flatten(),
          Dense(64, activation='relu'),
          Dense(1, activation='sigmoid'),
          Lambda(lambda x: x, output_shape=lambda s: s)  # Aggiungi questo layer Lambda
    ])
      elif string_model == 'conv2d':
        self.model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.df.shape[1], self.df.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(self.df.shape[1] * self.df.shape[2], activation='sigmoid'),
        Reshape((self.df.shape[1], self.df.shape[2]))  # Modifica qui la dimensione dell'output
    ])
      elif string_model == 'conv3d':
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
        self.model = keras.Sequential([
          GRU(64, input_shape=((self.df.shape[1]), 1), return_sequences=True),
          GRU(32),
          Dense((self.df.shape[1]), activation='linear'),# Modifica qui la dimensione dell'output
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'GRU2D':
        self.model = keras.Sequential([
          GRU(64, input_shape=((self.df.shape[1:])), return_sequences=True),
          GRU(32),
          Dense(self.tuple_prod(self.df.shape[1:]), activation='linear'),# Modifica qui la dimensione dell'output
          Reshape((self.df.shape[1:])),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM1D':
        self.model = keras.Sequential([
          LSTM(64, input_shape=((self.df.shape[1]), 1), return_sequences=True),
          LSTM(32),
          Dense((self.df.shape[1]), activation='linear'),# Modifica qui la dimensione dell'output
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM2D':
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



  def detect_deep_anomalies(self):
    reconstructed = self.model.predict(self.df)
    mse = np.mean(np.power(self.df - reconstructed, 2), axis=1)
    rate=[]
    parameters = np.linspace(mse.min(), mse.max(), 1000)
    for i in range(len(parameters)):
      anomal = np.where(mse > parameters[i])

      if len(self.df.shape) == 2:
        rate.append((len(anomal[0]))/(self.df.shape[0]))
      elif len(self.df.shape) == 3:
        rate.append((len(anomal[0]))/(self.df.shape[0]*self.df.shape[1]))
      elif len(self.df.shape) == 4:
        rate.append((len(anomal[0]))/(self.df.shape[0]*self.df.shape[1]*self.df.shape[2]))        

    sns.set_style('darkgrid')
    plt.plot(parameters, rate)
    plt.scatter(parameters, rate)
    plt.xlabel('mse threshold')
    plt.ylabel('anomaly rate ')
    plt.show()

    der = np.gradient(rate)

    ind_der = np.argmax(der)

        # Plot della derivata
    plt.plot(parameters, der)
    plt.xlabel('mse threshold')
    plt.ylabel('derivative')
    plt.show()

    print("Indice della derivata massima:", ind_der)
    print("Valore in quel punto:", parameters[ind_der])

    anomalies_indices=[]
    
    '''print((mse.shape))
    for i in range(len(mse)):
       for j in range(len(mse[i])):
        if(mse[i][j]>rate[ind_der]):
            anomalies_indices.append([i,j])'''
    
    anomalies_indices = np.where(mse > parameters[ind_der])
    print('Gli indici delle anomalie risultano essere:', anomalies_indices)
    return(anomalies_indices)
  
  def KMeans_anomalies(self):
    distances = self.model.transform(self.df)

    # 3. Calcolo della soglia
    mean_distance = np.mean(np.min(distances, axis=1))
    std_distance = np.std(np.min(distances, axis=1))

    threshold = mean_distance + 2 * std_distance  # Esempio: soglia come due deviazioni standard sopra la media

    # 4. Identificazione delle anomalie
    anomaly_indices = np.where(np.min(distances, axis=1) > threshold)[0]
    anomalies = self.df[anomaly_indices]

    print("Indici delle anomalie:", anomaly_indices)


  def forest_svm_anomalies(self):
    anomaly_scores = self.model.decision_function(self.df)  # Calcolo degli score di anomalia per il dataset di test

# 3. Definizione della soglia -8.5 per SVM
    threshold = -8.5  # Esempio: definizione della soglia per identificare le anomalie

# 4. Identificazione delle anomalie
    anomaly_indices = np.where(anomaly_scores < threshold)[0]
    anomalies = self.df[anomaly_indices]

    print("Indici delle anomalie:", len(anomaly_indices))

  def lof_anomalies(self):

    # Calcolo degli score di anomalia per il dataset di tes
    anomalies = self.model.fit_predict(self.df)
    anomaly_indices = np.where(anomalies < 0)[0]
    print("Indici delle anomalie:", anomaly_indices)

