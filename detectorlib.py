import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
import keras
import keras.layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Reshape, LSTM, GRU, Conv1D, Conv3D, MaxPooling1D, MaxPooling3D, Conv1DTranspose
from sklearn.neighbors import LocalOutlierFactor
from datetime import timedelta
from matplotlib.colors import Normalize
from plot_keras_history import plot_history
import tensorflow as tf
'''In this class we have to set the local variables to assign ath every index of our notations'''
class detector():

  '''                                                                                        
Explained variance for shape [41]_[11, 16, 10]: 0.8681266505022884
Explained variance for shape [41, 11]_[16, 10]: 0.8632755943305828
Explained variance for shape [41, 11, 10]_[16]: 0.9531381520347932
Explained variance for shape [41, 11, 16]_[10]: 0.9067262065543965
Explained variance for shape [16, 10]_[41, 11]: 0.3582718491728941
Explained variance for shape [10]_[41, 11, 16]: 0.8149863586217296
Explained variance for shape [16]_[10, 11, 41]: 0.6690677713468111
'''
  
  def tuple_prod(self, tuple):
    prod = 1
    for dim in tuple:
        prod *= dim
    return prod
  

  '''Reads from excel file the data and append the sheets to the third index of the tensor: (temporal samples, features, sensors)'''
  def load_preprocess(self, path, sens_num):
        self.df = []
        self.xlsx_path=path
        for sheet_num in range(sens_num): 
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)
            self.timestamp = sheet_df['timestamp'].dt.date
            sheet_df = sheet_df.drop(['Unnamed: 0', 'timestamp', 'sensor', 'off_ch1', 'off_ch2', 'off_ch3', 'off_ch4'], axis=1)
            self.df.append(sheet_df)
            
        
        self.df = np.array(self.df)
        self.df = np.nan_to_num(self.df)
        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        self.df = self.df.transpose(1, 0, 2)


  '''Given the desired index from the main, it reshape the df into a tensor as the user wants'''
  def reshape_ortogonal_tensor(self, temporal_indices, spatial_indices):
    if temporal_indices[0] == 0 and temporal_indices[1] == 0:
        print('Please set at least one temporal index different from 0')
        return
    
    if spatial_indices[0] == 0 and spatial_indices[1] == 0 and spatial_indices[2] == 0:
        print('Please set at least one spatial index different from 0')
        return
    
    new_temporal_indices = [x for x in temporal_indices if x != 0]
    new_spatial_indices = [x for x in spatial_indices if x != 0]

    
    if 0 not in temporal_indices and 0 not in spatial_indices:
      self.df = self.df.reshape(temporal_indices[0], temporal_indices[1], spatial_indices[0], spatial_indices[1], spatial_indices[2])
    else:
      self.df = self.df.reshape(*new_temporal_indices, *new_spatial_indices)
  
  
  '''You have to pass the indices without zeros'''
  # devi cambiare tutte le volte la shape da cui parti per essere sicuro
  def reshape_linear_tensor(self, temporal_indices, spatial_indices, standardize=False):

    self.temporal_indices = temporal_indices
    self.spatial_indices = spatial_indices
    self.df = np.array(self.df.reshape(self.tuple_prod(temporal_indices), self.tuple_prod(spatial_indices)))
    if standardize == True:

      mean = np.mean(self.df, axis=0)
      std_dev = np.std(self.df, axis=0)
      self.df = (self.df - mean)/(std_dev)
      self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
          

  def create_statistical_model(self, string_model):
      try:
        if string_model == 'KMeans':
            self.str_model=string_model
            self.model = KMeans(n_clusters=5) 
        elif string_model == 'IsolationForest':
            self.str_model=string_model
            self.model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.2), random_state=42) 
        elif string_model == 'SVM':
            self.str_model=string_model
            self.model = OneClassSVM(gamma='auto')  
        elif string_model == 'LOF':
            self.str_model=string_model
            self.model = LocalOutlierFactor() 
        elif string_model == 'PCA':
            self.str_model=string_model
            self.model = PCA(n_components=4)
        else:
            raise ValueError('Model name not recognized')
      except ValueError as e:
        print(f"Error creating model: {e}")
        return None

  def create_PCA(self):
    explained_variances=[]

    for i in range(10):
      pca = PCA(n_components=i+1)
      pca.fit(self.df)
      explained_variances.append(sum(pca.explained_variance_ratio_))
    self.PCA_Ncomponents = next((i for i, valore in enumerate(explained_variances) if valore > self.variance_components), len(explained_variances) - 1)+1
    self.model=PCA(n_components=self.PCA_Ncomponents)
    return


  def hyperopt_deep_model(self, string_model):
      
    self.str_model = string_model + f"_layers{self.layers}_KerStrd_{self.ker_strd}_nodes_64"
    self.model = keras.Sequential()
    for i in range(self.layers):
        self.model.add(Conv1D(filters=4, kernel_size=self.ker_strd, strides=self.ker_strd, activation='relu', padding='same', input_shape=(self.tuple_prod(self.spatial_indices), 1)))
    self.model.add(Flatten())
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dense((self.df.shape[1]), activation='linear'))
    self.model.add(Lambda(lambda x: x, output_shape=lambda s: s))
     

  
  def suitable_conv(self, filters, ker_str, dense, input):
        # Convolution
        encoded = tf.keras.layers.Conv1D(filters=filters, kernel_size=ker_str, strides=ker_str, activation='relu', padding='same')(input)
        # Downsampling
        encoded_fc1 = tf.keras.layers.Dense(dense, activation='relu')(encoded)
        encoded_fc2 = tf.keras.layers.Dense(dense, activation='relu')(encoded_fc1)
        # Deconvolution    
        decoded = tf.keras.layers.Conv1DTranspose(1, kernel_size=ker_str, strides=ker_str,  activation='relu', padding='same')(encoded_fc2)
        return decoded

  def create_deep_model(self, string_model):
    try:
      if string_model == 'conv1d':
        self.str_model=string_model + f"_nodes_20"
        input= [self.tuple_prod(self.spatial_indices), 1]
        input_img = keras.layers.Input(shape=input)
        if len(self.spatial_indices) == 1:
          self.model = keras.models.Model(input_img, self.suitable_conv(filters=2, ker_str=int(self.spatial_indices[0]/2), dense=20, input=input_img))
        else:
          self.model = keras.models.Model(input_img, self.suitable_conv(filters=2, ker_str=self.spatial_indices[0], dense=20, input=input_img))
        

        
      elif string_model == 'conv2d':
        self.str_model=string_model
        self.model = keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.df.shape[1], self.df.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(self.df.shape[1] * self.df.shape[2], activation='sigmoid'),
        Reshape((self.df.shape[1], self.df.shape[2])) 
    ])
        
      elif string_model == 'conv3d':
        self.str_model=string_model
        self.model = keras.Sequential([
          Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(self.df.shape[1], self.df.shape[2], self.df.shape[3], 1)),
          MaxPooling3D((2, 2, 2)),
          Flatten(),
          Dense(64, activation='relu'),
    
          Dense(self.df.shape[1]*self.df.shape[2]*self.df.shape[3], activation='sigmoid'),
          Reshape((self.df.shape[1], self.df.shape[2], self.df.shape[3])),
          Lambda(lambda x: x, output_shape=lambda s: s)  
    ])
        
      elif string_model == 'GRU1D':
        self.str_model=string_model+f"_16_32"
        self.model = keras.Sequential([
          GRU(16, input_shape=(self.tuple_prod(self.spatial_indices), 1), return_sequences=True),
          GRU(32),
          Dense((self.df.shape[1]), activation='linear'),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'GRU2D':
        self.str_model=string_model
        self.model = keras.Sequential([
          GRU(64, input_shape=((self.df.shape[1:])), return_sequences=True),
          GRU(32),
          Dense(self.tuple_prod(self.df.shape[1:]), activation='linear'),
          Reshape((self.df.shape[1:])),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM1D':
        self.str_model=string_model
        self.model = keras.Sequential([
          LSTM(64, input_shape=(self.tuple_prod(self.spatial_indices), 1), return_sequences=True),
          LSTM(32),
          Dense((self.df.shape[1]), activation='linear'),
          Lambda(lambda x: x, output_shape=lambda s: s)
    ])
        
      elif string_model == 'LSTM2D':
        self.str_model=string_model
        self.model = keras.Sequential([
          LSTM(64, input_shape=((self.df.shape[1:])), return_sequences=True),
          LSTM(32),
          Dense(self.tuple_prod(self.df.shape[1:]), activation='linear'),
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
    if self.tuple_prod(self.spatial_indices) < 160:
      history= self.model.fit(self.df, self.df, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
    else:
      history= self.model.fit(self.df, self.df, epochs=self.tuple_prod(self.spatial_indices), batch_size=32, validation_split=0.1, verbose=1)
    plot_history(history)
    plt.savefig(f'final_validation_autoencoder_{self.xlsx_path}/training_model_{self.str_model}_{self.temporal_indices}_{self.spatial_indices}.png')
    plt.close()


  def fit_model(self):
      self.model.fit(self.df)

  def fit_linear_model(self):
    self.model.fit(self.df, self.df)


  def deep_anomalies(self):
    self.fit_deep_model()
    reconstructed = (self.model.predict(self.df)).reshape(self.df.shape)
    print(reconstructed.shape)
    mse = np.mean(np.power(self.df - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 100 - 10)
    anomalies_idx = np.where(mse > threshold)[0]
    self.anomalies_indices = anomalies_idx[np.argsort(mse[anomalies_idx])[::-1]]


  def anomalies_stat(self):
    anomaly_percentage = 0.1  # 10%


    if self.str_model == 'SVM':
        self.fit_model() 
        anomaly_scores = self.model.decision_function(self.df)
        num_anomalies = int(anomaly_percentage * len(self.df))
        anomaly_indices = np.where(anomaly_scores < 0)[0]  
        sorted_anomaly_indices = np.argsort(anomaly_scores[anomaly_indices])
        selected_anomaly_indices = anomaly_indices[sorted_anomaly_indices[:num_anomalies]]
        self.anomalies_indices = selected_anomaly_indices

    elif self.str_model == 'KMeans':
        self.fit_model()  
        distances = self.model.transform(self.df)
        mean_distance = np.mean(np.min(distances, axis=1))
        std_distance = np.std(np.min(distances, axis=1))
        threshold = mean_distance + 2 * std_distance  
        self.anomalies_indices = np.where(np.min(distances, axis=1) > threshold)[0]
        num_anomalies = int(anomaly_percentage * len(self.df))
        sorted_anomalies_indices = np.argsort(np.min(distances, axis=1))[::-1]
        self.anomalies_indices = sorted_anomalies_indices[:num_anomalies]

    elif self.str_model == 'LOF':
        self.fit_model()  
        anomalies = self.model.fit_predict(self.df)
        self.anomalies_indices = np.where(anomalies < 0)[0]
        num_anomalies = int(anomaly_percentage * len(self.df))
        self.anomalies_indices = np.argsort(anomalies)[::-1][:num_anomalies]

    elif self.str_model == 'IsolationForest':
        self.fit_model()  
        anomaly_scores = self.model.decision_function(self.df)
        num_anomalies = int(anomaly_percentage * len(self.df))
        anomaly_indices = np.where(anomaly_scores < 0)[0]  
        sorted_anomaly_indices = np.argsort(anomaly_scores[anomaly_indices])
        selected_anomaly_indices = anomaly_indices[sorted_anomaly_indices[:num_anomalies]]
        self.anomalies_indices = selected_anomaly_indices

    elif self.str_model == 'PCA':
        self.fit_model()
        X_pca = self.model.transform(self.df)
        X_reconstructed = self.model.inverse_transform(X_pca)
        mse = np.mean(np.square(self.df - X_reconstructed), axis=1)
        threshold = np.percentile(mse, 100 - anomaly_percentage*100)
        anomaly_indices = np.where(mse > threshold)[0]
        self.anomalies_indices = anomaly_indices[np.argsort(-mse[anomaly_indices])]

    else:
        print("Unknown model")


  def save_linear_anomaly_indices(self):
      # divido l'indice dell'anomalia per uno degli indici temporali, poi trovo l'intero piu vicino e ho fatto teoricamente

    with open(f'final_validation_autoencoder_{self.xlsx_path}/anomalies_{self.str_model}_{self.temporal_indices}_{self.spatial_indices}.txt', 'w') as file:
        for indice in self.anomalies_indices:
          
          if len(self.temporal_indices) == 2:
            indice= int(indice/(self.temporal_indices[1])) +1, round(self.temporal_indices[1]*((indice/self.temporal_indices[1]) % 1)) +1

          elif len(self.temporal_indices) == 3:
            indice=round(self.temporal_indices[0]*((indice/(self.temporal_indices[0]*self.temporal_indices[1])) % 1)), int(self.temporal_indices[1]*((indice/(self.temporal_indices[1])) % 1)) +1, int(indice/(self.temporal_indices[1]*self.temporal_indices[0])) +1
            
          file.write(f"{indice}\n")


  def stamp_all_shape_anomalies(self, possible_shapes):
    variance_txt=[0.8681266505022884, 
                            0.8632755943305828, 
                            0.9531381520347932, 
                            0.9067262065543965, 
                            0.3582718491728941, 
                            0.8149863586217296, 
                            0.6690677713468111]
    colors = plt.get_cmap('tab10').colors
    
    i=0
    explained_graph_variances=[]
    plt.figure(figsize=(9,6))
    plt.rcParams.update({'font.size': 12})
    with open('final_variances_std_pca_2022.txt', 'a') as variances:
      for temporal_indices, spatial_indices in tqdm(possible_shapes, desc="Stamping shape anomalies"):
            self.reshape_linear_tensor(temporal_indices, spatial_indices, standardize=True)
            explained_graph_variances.append(self.PCA_graph())
            if self.str_model == 'PCA':
                if temporal_indices == [41]:
                    self.variance_components= variance_txt[0]
                elif temporal_indices == [41, 11]:
                    self.variance_components = variance_txt[1]
                elif temporal_indices == [41, 11, 10]:
                    self.variance_components = variance_txt[2]
                elif temporal_indices == [41, 11, 16]:
                    self.variance_component = variance_txt[3]
                elif temporal_indices == [16, 10]:
                    self.variance_components = variance_txt[4]
                elif temporal_indices == [10]:
                    self.variance_components = variance_txt[5]
                elif temporal_indices == [16]:
                    self.variance_components = variance_txt[6]
                self.create_PCA()
                self.fit_model()
                variances.write(f"Explained variance for shape {temporal_indices}_{spatial_indices}: {np.sum(self.model.explained_variance_ratio_)}\n")
            self.anomalies_stat()
            self.save_linear_anomaly_indices()

            
            plt.plot(np.arange(1, 11, 1), explained_graph_variances[i], label=f'{self.temporal_indices}_{self.spatial_indices}', color=colors[i])
            plt.scatter(np.arange(1, 11, 1), explained_graph_variances[i], edgecolors='black', color=colors[i])
            plt.axhline(y=variance_txt[i], linestyle='dashed', color=colors[i])
            plt.xlabel('components', fontsize=15)
            plt.ylabel('variance explained', fontsize=15)
            plt.legend()


            i=i+1
    plt.savefig(f'All_variances_in_a_graph')
    plt.close()
    self.PCA_graph()


  
  def stamp_all_shape_deep_anomalies(self, possible_shapes, model):

      for temporal_indices, spatial_indices in tqdm(possible_shapes, desc="Stamping shape anomalies"):
        self.reshape_linear_tensor(temporal_indices, spatial_indices)
        self.create_deep_model(model)
        self.deep_anomalies()
        self.save_linear_anomaly_indices()



  def hyperopt_anomalies(self, possible_shapes, model, possible_combinations):
    for combinations in tqdm(possible_combinations, desc="Stamping hyperpatameters"):
      ker_strd, layers = combinations
      self.ker_strd = ker_strd
      self.layers = layers
      for temporal_indices, spatial_indices in tqdm(possible_shapes, desc="Stamping shape anomalies"):
        self.reshape_linear_tensor(temporal_indices, spatial_indices)
        self.hyperopt_deep_model(model)
        self.deep_anomalies()
        self.save_linear_anomaly_indices()

     

  def PCA_graph(self):
    sns.set_style('darkgrid')

    if self.temporal_indices == [16, 20]:
        n_components_range = range(1, 51)
    else:
        n_components_range = range(1, 11)

    
    explained_variances = []

    for n_components in n_components_range:
        pca = PCA(n_components=n_components)
        pca.fit(self.df)
        explained_variances.append(sum(pca.explained_variance_ratio_))

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, explained_variances, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Total Variance Explained')
    plt.title('Total Variance Explained by Number of Components')
    plt.axhline(y=0.95, linestyle='dashed', color='red')
    plt.savefig(f'graphs_variance_PCA {self.xlsx_path}/shape_{self.temporal_indices}_{self.spatial_indices}')
    plt.close()

    self.PCA_Ncomponents = next((i for i, valore in enumerate(explained_variances) if valore > 0.95), len(explained_variances) - 1)+1
    print(self.PCA_Ncomponents, self.temporal_indices, self.spatial_indices)

    pca = PCA(n_components=self.PCA_Ncomponents)
    pca.fit(self.df)
    scores = pca.transform(self.df)
    # devo calcolare i loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    t_squared = np.sum((scores[:, :self.PCA_Ncomponents] / np.sqrt(pca.explained_variance_))**2, axis=1)
    q_residuals = np.sum(self.df**2 - np.dot(scores, loadings.T)**2, axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(22, 8))
    plt.rcParams.update({'font.size': 15})
    axs[0].scatter(scores[:, 0], scores[:, 1], color='lightblue', edgecolors='black')
    axs[0].set_xlabel(f'PC1: {round(explained_variances[0], 3)*100}%')
    axs[0].set_ylabel(f'PC2: {round(explained_variances[1]-explained_variances[0], 3)*100}%')
    axs[0].set_xlim([-4, 4])
    axs[0].set_ylim([-4, 4])
    axs[0].axhline(y=0, linestyle='dashed', color='red')
    axs[0].axvline(x=0, linestyle='dashed', color='red')
    axs[0].set_title('Scores Plot')

    # ONly for the article

    if self.temporal_indices == [16, 10]:
      axs[1].plot((range(451)), loadings[:, 0], label='PC1')
      axs[1].plot((range(451)), loadings[:, 1], label='PC2')
      axs[1].set_xticks(ticks=(range(451)[::150]), labels=self.timestamp[::150], rotation=20)
      axs[1].set_xlabel('Time')


    else:
      axs[1].plot(range(self.tuple_prod(self.spatial_indices)), loadings[:, 0], label='PC1')
      axs[1].plot(range(self.tuple_prod(self.spatial_indices)), loadings[:, 1], label='PC2')
      axs[1].set_xlabel('Loadings')
    axs[1].set_ylabel('Variables')
    axs[1].axhline(y=0, linestyle='dashed', color='red')
    axs[1].axvline(x=0, linestyle='dashed', color='red')
    axs[1].set_title('Loadings Plot')
    axs[1].legend()

    axs[2].scatter(t_squared, q_residuals, color='lightblue', edgecolors='black')
    axs[2].axhline(y=np.max(q_residuals)*0.95, color='red', linestyle='--', label='Q Residuals 95% Threshold')
    axs[2].axvline(x=np.max(t_squared)*0.95, color='red', linestyle='--', label='T-squared 95% Threshold')
    axs[2].set_xlabel('T-squared (t^2)')
    axs[2].set_ylabel('Q Residuals')
    axs[2].set_title('T-squared vs Q Residuals')
    axs[2].legend()

    plt.savefig(f'graphs_variance_PCA {self.xlsx_path}/scores_vs_loadings_shape_{self.temporal_indices}_{self.spatial_indices}')
    plt.tight_layout()
    plt.close()
    return explained_variances
    


class sheet:
    def __init__(self):
        pass

    def load_timestamps(self, path, sens_num):
        self.df = []
        for sheet_num in range(sens_num):  # Change to range(18) when you have all
            sheet_df = pd.read_excel(path, sheet_name=sheet_num)
            sheet_df = sheet_df['Date Acquisition']
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
        
                if curr_date != prev_date + timedelta(days=1) and curr_date != prev_date + timedelta(days=0):
                    print(f"Discontinuity found on array {idx} on date : {prev_date}")
                    print(f"Delta days: {abs(curr_date - prev_date).days}")
                    discs.append(idx)
                    dates.append(prev_date)
                prev_date = curr_date
            
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
        plt.rcParams.update({'font.size': 15})
        for idx, col in enumerate(self.df[i].iloc[:, 2:].columns):
            color = cmap(normalize(idx))
            plt.plot(self.df[i]['timestamp'], self.df[i][col], label=col, color=color)
        plt.title(self.df[i]['sensor'].iloc[0])
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Intensity', fontsize=20)
        plt.legend()
        plt.savefig(f'graphs {self.xlsx_path}/sensor_{self.df[i]["sensor"].iloc[0]}.png')
        plt.close()