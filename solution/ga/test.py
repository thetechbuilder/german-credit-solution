#! /usr/bin/env python3
import unittest
import test_ga

def load_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(
        test_ga.TestGaBaseClasses))
    return suite

unittest.TextTestRunner(verbosity = 2).run(load_suite())
