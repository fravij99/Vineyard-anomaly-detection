import unittest
import detectorlib
import numpy as np

path="2021_All raw data_No_background.xlsx"
det=detectorlib.detector()
det.load_preprocess(path, 4)

class TestDetector(unittest.TestCase):

    def test_reshape_tensor(self):
        
        det.reshape_tensor([693, 0], [0, 16*4, 0])
        self.assertEqual(det.df.shape, (693, 16* 4))
        det.reshape_tensor([63, 11], [16, 4, 0])
        self.assertEqual(det.df.shape, (63, 11, 16, 4))
        det.reshape_tensor([693, 0], [16, 4, 0])
        self.assertEqual(det.df.shape, (693, 16, 4))

    def test_create_model(self):
        det.create_model('KMeans')
        self.assertEqual(str(det.model), 'KMeans(n_clusters=5)')
        det.create_model('LOF')
        self.assertEqual(str(det.model), 'LocalOutlierFactor()')
        det.create_model('IsolationForest')
        self.assertEqual(str(det.model), 'IsolationForest(contamination=0.2, random_state=42)')
        det.create_model('SVM')
        self.assertEqual(str(det.model), ("OneClassSVM(gamma='auto')"))   
        det.create_model('linear')
        self.assertEqual(str(det.model), ("LinearRegression()"))   


    def test_create_deep_model(self):
        
        # Test 'conv1d'
        det.reshape_tensor([693, 0], [0, 16*4, 0])
        det.create_deep_model('conv1d')
        self.assertIsNotNone(det.model)

        # Test 'conv2d'
        det.reshape_tensor([693, 0], [16, 4, 0])
        det.create_deep_model('conv2d')
        self.assertIsNotNone(det.model)

        # Test 'conv3d'
        det.reshape_tensor([63, 11], [16, 4, 0])
        det.create_deep_model('conv3d')
        self.assertIsNotNone(det.model)

        # Test 'GRU1D'
        det.reshape_tensor([693, 0], [0, 16*4, 0])
        det.create_deep_model('GRU1D')
        self.assertIsNotNone(det.model)

        # Test 'GRU2D'
        det.reshape_tensor([693, 0], [16, 4, 0])
        det.create_deep_model('GRU2D')
        self.assertIsNotNone(det.model)

        # Test 'LSTM1D'
        det.reshape_tensor([693, 0], [0, 16*4, 0])
        det.create_deep_model('LSTM1D')
        self.assertIsNotNone(det.model)

        # Test 'LSTM2D'
        det.reshape_tensor([693, 0], [16, 4, 0])
        det.create_deep_model('LSTM2D')
        self.assertIsNotNone(det.model)

    def test_find_discontinuity(self):
        timestamps1 = np.array(['18/09/2021 01:11:02', '19/09/2021 02:22:02', '20/09/2021 03:33:02'])
        timestamps2 = np.array(['17/09/2021 04:44:02', '18/09/2021 05:11:02', '19/09/2021 06:11:02', '19/09/2021 07:12:02'])
        timestamps3 = np.array(['17/09/2021 07:13:02', '18/09/2021 08:12:02', '19/09/2021 09:11:02'])
        timestamps4 = np.array(['17/09/2021 07:11:02', '18/09/2021 08:18:02', '21/09/2021 09:55:02'])

        sh=detectorlib.sheet()
        all= sh.find_discontinuity(timestamps1, timestamps2, timestamps3, timestamps4)
        print(all)
        self.assertEqual(str(all), '([3], [datetime.date(2021, 9, 18)])')

if __name__ == "__main__":
    unittest.main()