#!/usr/bin/env python

import unittest
import nbconvert
import os

import numpy as np

with open("assignment8.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)


with open("assignment8.py", "w") as f:
    f.write(python_file)


from assignment8 import KozenyCarmen

class TestSolution(unittest.TestCase):

    def test_transform(self):

        kc = KozenyCarmen('poro_perm.csv')

        np.testing.assert_allclose(kc.kc_model()[0:10], 
                                   np.array([0.00144518, 0.00144518, 0.00178167, 
                                             0.00073352, 0.0035369, 0.00123457, 
                                             0.00194181, 0.00199742, 0.0022314, 
                                             0.00205417]), atol=0.0001)
        
if __name__ == '__main__':
    unittest.main()

