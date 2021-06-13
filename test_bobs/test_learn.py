import os
import unittest

import sklearn.linear_model as lm
from sklearn import dummy
import numpy as np

from bobs import learn


class TestLearn(unittest.TestCase):
    def test_load_or_train_file_absent(self):
        path = 'linear.pkl'
        try:
            os.remove(path)
        except OSError:
            pass
            
        trained_model = learn.load_or_train(path, self.train)
        pred = trained_model.predict(np.array([[5, 6], [7, 8]]))
        
        self.assertEqual(len(pred), 2)
        for pred_e, actual_e in zip(pred, [11, 15]):
            self.assertAlmostEqual(pred_e, actual_e)
        self.assertTrue(os.path.exists(path))
    
    def test_load_or_train_file_present(self):
        path = 'dummy.pkl'
        dummy_model = dummy.DummyRegressor(strategy="constant", constant=42)
        dummy_model.fit(np.array([42]), np.array([42]))
        learn.save(path, dummy_model, method="pickle")
            
        trained_model = learn.load_or_train(path, self.train)
        pred = trained_model.predict(np.array([[5, 6], [7, 8]]))
        
        self.assertEqual(list(pred), [42, 42])
    
    @staticmethod
    def train():
        x = np.array([[1, 2], [3, 4]])
        y = np.array([3, 7])
        model = lm.LinearRegression()
        model.fit(x, y)
        return model

