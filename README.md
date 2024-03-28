# README - Anomaly Detection Library
Hey there! 

This is my master thesis repository. 
This library provides tools for anomaly detection in multi-dimensional datasets came form iGrape projects. It includes methods for data preprocessing, model creation, and anomaly detection using various algorithms.

## Usage
### Loading and Preprocessing Data

```
det = detector()
det.load_preprocess('path_to_file.xlsx', sens_num)

```
`load_preprocess(path, sens_num)`: Reads data from an Excel file and preprocesses it. 

`sens_num` specifies the number of sheets to load.

### Reshaping Data
```
det.reshape_tensor(temporal_indices, spatial_indices)
```

`reshape_tensor(temporal_indices, spatial_indices)`: Reshapes the data tensor based on temporal and spatial indices.
### Creating Models

```
det.create_model(string_model)
det.create_deep_model(string_model)
create_model(string_model): Creates models for anomaly detection.
```

`create_deep_model(string_model)`: Creates deep learning models for anomaly detection.

Supported `string_model options`:

* 'KMeans'
* 'IsolationForest'
* 'SVM'
* 'LOF'
* 'conv1d'
* 'conv2d'
* 'conv3d'
* 'GRU1D'
* 'GRU2D'
* 'LSTM1D'
* 'LSTM2D'


### Fitting Models

```
det.fit_model()
det.fit_deep_model()
det.fit_ridge()
```

`fit_model()`: Fits the selected statistical model to the data.

`fit_deep_model()`: Fits deep learning models to the data.

`fit_linear()`: Fits a Linear Regressor model to the data.


### Anomaly Detection

```
det.detect_deep_anomalies()
det.KMeans_anomalies()
det.forest_svm_anomalies()
det.lof_anomalies()
```

`detect_deep_anomalies()`: Detects anomalies using deep learning models.

`KMeans_anomalies()`: Detects anomalies using KMeans clustering.

`forest_svm_anomalies()`: Detects anomalies using SVM or Isolation Forest.

`lof_anomalies()`: Detects anomalies using Local Outlier Factor.

## Example
```
from anomaly_detection import detector

# Create instance of detector
det = detector()

# Load and preprocess data
det.load_preprocess('data.xlsx', sens_num)

# Reshape data tensor
det.reshape_tensor(temporal_indices=[10, 20], spatial_indices=[5, 5, 5])

# Create and fit model
det.create_model('SVM')
det.fit_model()

# Detect anomalies
det.forest_svm_anomalies()
```

## Installation
You can clone the repository using the command:

```
git clone git@github.com:fravij99/Vineyard-anomaly-detection.git
```

## Requirements
* Python 3.6+
* TensorFlow
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn


For deep learning models, you may need additional packages such as Keras.

## License
This library is provided under the MIT License.

## Contributions
Contributions and feedback are welcome! Please feel free to open issues or submit pull requests.

## Author
Developed by [Francesco Villa][fravi]

## Contact
For questions or support, contact [fravilla30@gmail.com] or [francesco.villa6@studenti.unimi.it].

[fravi]: https://github.com/fravij99
