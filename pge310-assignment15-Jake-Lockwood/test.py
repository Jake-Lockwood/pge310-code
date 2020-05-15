#!/usr/bin/env python

import unittest

from unittest.mock import MagicMock

import nbconvert

import numpy as np


with open("assignment15.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)

with open("assignment15.py", "w") as f:
    f.write(python_file)

from assignment15 import IterativeSolver


class TestSolution(unittest.TestCase):

    def test_solver(self):
        A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        b = np.array([1, 2, 3])
        solver = IterativeSolver(A, b)
        np.testing.assert_array_almost_equal(solver.gauss_seidel_solve(), np.array([2.5, 4., 3.5]), decimal=6)

#        
if __name__ == '__main__':
    unittest.main()
