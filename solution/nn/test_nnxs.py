#! /usr/bin/env python3
#
#test_nnxt.py
#
"""Unit tests for nnxs.py"""

#standard modules:
import os
import unittest
from random import randint, random
#included modules:
from neuro import Network
from nnxs import NNXSerializer

test_file = "nnxs_test_file.xml"

class TestNNXSerializer(unittest.TestCase):
    def setUp(self):
        #prepare the test fixture:
        self.nn = Network(*[randint(1, 10) 
            for x in range(randint(2, 10))], n = random())
    
    def tearDown(self):
        os.remove(test_file)
    
    def test_read_and_write_1(self):
        NNXSerializer.write(self.nn, test_file)
        result = NNXSerializer.parse(test_file)
        self.__check(result)

    def test_read_and_write_2(self):
        with open(test_file, "wb") as f:
            NNXSerializer.write(self.nn, f)
        with open(test_file, "rb")as f:
            result = NNXSerializer.parse(f)
        self.__check(result)

    def test_error_handling(self):
        with open(test_file, "w") as f:
            self.assertRaises(TypeError, NNXSerializer.write, self.nn, f)
        
        NNXSerializer.write(self.nn, test_file)
        with open(test_file, "r") as f:
            self.assertRaises(TypeError, NNXSerializer.parse, f)
        self.assertRaises(TypeError, NNXSerializer.write, 
                random(), test_file)
    
    def __check(self, result):
        self.assertEqual(result.speed, self.nn.speed)
        self.assertEqual(result.algorithm, self.nn.algorithm)
        
        self.assertSequenceEqual(result.shape, self.nn.shape)
        self.assertSequenceEqual(result._links, self.nn._links)

#######################################################################
### Run tests
#######################################################################

def _test():
    unittest.main(verbosity = 2)

#run test
if __name__ == '__main__':
    _test()
