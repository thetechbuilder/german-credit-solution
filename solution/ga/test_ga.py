from unittest import TestCase
from ga import *
from random import randint

class TestGaBaseClasses(TestCase):
    def testIndividual(self):
        individual = GaIndividualBase()
        individual.fitness = 10
        self.assertEqual(individual.fitness, 10)
    
    def testGaSeqBase(self):
        self.assertRaises(TypeError, GaSeqBase)
        self.assertRaises(TypeError, GaBase)

    def testGaOperators_1(self):
        ops = lambda x: x*x, lambda x: x + 1
        weights = 10, 20

        gops = GaOperators(ops, weights)
        #__getitem__ test:
        self.assertEqual((gops[0], gops[1]), tuple(zip(ops, weights)))

        #repeated function test
        gops = GaOperators((ops[0], ops[0]), weights)
        self.assertEqual(len(gops), 1)
        gops.append((ops[0], 203))
        self.assertEqual(len(gops), 1)

        #del item test:
        del gops[0]
        self.assertEqual(len(gops), 0)

        #exceptions:
        self.assertRaises(IndexError, gops.__getitem__, 99)
        self.assertRaises(TypeError, gops.__getitem__, 'str')
        self.assertRaises(ValueError, gops.append, (',', -2))
        self.assertRaises(ValueError, gops.append, ('k', 'j'))

    def testGaOperators_2(self):
        #tests how weights are assinging by GaOperators
        calls = 9
        length = 10 #must be less than call count
        weights = [randint(1, 10) for x in range(length)]
        operators = list(range(length))
        G = GaOperators(operators, weights)

        for x in range(calls):
            #checking recalculation of weights while deleting and chainging
            del_i = randint(0, length - 1)
            cng_i = randint(0, length - 2)
            cng_w = randint(1, 10)
            del G[del_i]
            G[cng_i] = 'test-operator', cng_w
            del weights[del_i]
            weights[cng_i] = cng_w

            length -= 1
            #checking all weights:
            w = 0
            for i in range(length):
                w+=weights[i]
                self.assertEqual(G._GaOperators__weights[i], 
                        w, "Wrong assignation of weights")
            
            #test choosing
            G.select()
    
    def testSlices(self):
        calls = 10
        length = 10
        for x in range(calls):
            weights = [randint(1, 10) for x in range(length)]
            operators = list(range(length)) #test operators
            gopr = GaOperators(operators, weights)
            slc = slice(randint(0, length - 1), randint(0, length - 1))
            self.assertEqual(gopr[:], list(zip(operators, weights)))
            self.assertEqual(gopr[slc], 
                    list(zip(operators[slc], weights[slc])))
            
            #slice chainging
            new_vals = [randint(1, 100) 
                    for x in range(*slc.indices(length))]
            gopr[slc] = zip(operators[slc], new_vals)
            weights[slc] = new_vals
            self.assertEqual(gopr[:], list(zip(operators, weights)))
