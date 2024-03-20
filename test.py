import unittest
import detectorlib
import numpy as np

path="2021_All raw data_No_background.xlsx"
det=detectorlib.detector()
det.load_preprocess(path, 5)

class TestDetector(unittest.TestCase):

    def test_reshape_tensor(self):
        
        det.reshape_tensor([693, 0], [0, 17*4, 0])
        self.assertEqual(det.df.shape, (693, 17* 4))
        det.reshape_tensor([63, 11], [17, 4, 0])
        self.assertEqual(det.df.shape, (63, 11, 17, 4))
        det.reshape_tensor([693, 0], [17, 4, 0])
        self.assertEqual(det.df.shape, (693, 17, 4))

    def test_create_model(self):
        det.create_model('KMeans')
        self.assertEqual(str(det.model), 'KMeans(n_clusters=5)')
        det.create_model('LOF')
        self.assertEqual(str(det.model), 'LocalOutlierFactor()')
        det.create_model('IsolationForest')
        self.assertEqual(str(det.model), 'IsolationForest(contamination=0.2, random_state=42)')
        det.create_model('SVM')
        self.assertEqual(str(det.model), ("OneClassSVM(gamma='auto')"))


    def test_create_deep_model(self):
        # Test 'conv1d'
        det.reshape_tensor([693, 0], [0, 17*4, 0])
        det.create_deep_model('conv1d')
        self.assertIsNotNone(det.model)


        # Test 'conv2d'
        det.reshape_tensor([693, 0], [17, 4, 0])
        det.create_deep_model('conv2d')
        self.assertIsNotNone(det.model)

        # Test 'conv3d'
        det.reshape_tensor([63, 11], [17, 4, 0])
        det.create_deep_model('conv3d')
        self.assertIsNotNone(det.model)

        # Test 'GRU1D'
        det.reshape_tensor([693, 0], [0, 17*4, 0])
        det.create_deep_model('GRU1D')
        self.assertIsNotNone(det.model)

        # Test 'GRU2D'
        det.reshape_tensor([693, 0], [17, 4, 0])
        det.create_deep_model('GRU2D')
        self.assertIsNotNone(det.model)

        # Test 'LSTM1D'
        det.reshape_tensor([693, 0], [0, 17*4, 0])
        det.create_deep_model('LSTM1D')
        self.assertIsNotNone(det.model)

        # Test 'LSTM2D'
        det.reshape_tensor([693, 0], [17, 4, 0])
        det.create_deep_model('LSTM2D')
        self.assertIsNotNone(det.model)





if __name__ == "__main__":
    unittest.main()