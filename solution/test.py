#! /usr/bin/env python3
#appending folders:
from sys import path
path.append('nn')
#standard modules:
import unittest
#included modules:
import test_neuro
import test_german_credit

def load_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        test_neuro.TestNeuralNetwork))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        test_neuro.TestMutableNeuralNetwork))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        test_german_credit.TestGermanCredit))
    return suite

unittest.TextTestRunner(verbosity = 2).run(load_suite())
