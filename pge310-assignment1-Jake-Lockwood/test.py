#/usr/bin/env python

import unittest

class TestSolution(unittest.TestCase):

    def test_solution(self):

        with open('solution.txt', 'r') as f:

            first_line = f.readline()

            assert first_line.rstrip() == "Hello, World!"


if __name__ == '__main__':
    unittest.main()
