#! /usr/bin/env python3
#
#test_neuro.py
#
"""Unit tests for neuro.py"""

#standard modules:
import math
import unittest
from random import randrange, randint, choice, uniform
from numbers import Number as _Number
from collections.abc import Sequence as _Sequence
#included modules:
from neuro import *
from pyextra import Progress

#global parameters:
_NUM = 50 #number of invokes per test
_SIZE = 20 #size coefficient that is describes how big test items must be
#notive: the size must be at least 3, ERROR othewise

mulsum = lambda W, O: sum(map(lambda o, w: o*w, O, W))
sigmoid = lambda x: 1/(1 + math.e**-x)
out = lambda W, O: sigmoid(mulsum(W[:-1], O) + W[-1])


class TestNeuralNetwork(unittest.TestCase):
    def ftest_network_structure(self):
        setup = i, j, k = 5, 2, 1
        # | o |  |   |  |   |
        # | o |  | o |  |   |
        # | o |  |   |  | o | } NEURONS
        # | o |  | o |  |   |
        # | o |  |   |  |   |
        # ___________________
        #   I      J      K
        #
        # So there are two sets of weighs: the weighs from layer i to j 
        # and from k to k respectively
        N = Network(setup)
        self.assertEqual(len(N[:]), 2, 
                'wrong structure: the number of weights is wrong')
        self.assertEqual(len(N[0] + N[1]), j + k,
                'wrong structure: the number of neurons is wrong')
        self.assertEqual(len(N[0][0]) + len(N[0][1]), 12, 
                'wrong struncture: the number the weighs from I to J '
                'is wrong')
        self.assertEqual(len(N[1][0]), 3, 
                'wrong struncture: the number the weighs ' 
                'from J to K is wrong')

    def ftest_network_evaluation(self):
        W = [random() for x in range(12)]
        setup = 5, 2, 1
        ins = [1, 2, 3, 4, 5]
        N = Network(setup)
        bias = 0.05
        N[0][0][:5], N[0][0][-1] = W[:5], bias
        N[0][1][:5], N[0][1][-1] = W[5:10], bias
        N[1][0][:2], N[1][0][-1] = W[10:], bias
        #expected result:
        o1 = out(W[:5] + [bias], ins)
        o2 = out(W[5:10] + [bias], ins)
        oo = sigmoid(o1*W[10] + o2*W[11] + bias)
        #neural network structure validating:
        self.assertTrue((N[0][0][:-1] == W[:5] and 
            N[0][1][:-1] == W[5:10] and 
                N[1][0][:-1] == W[10:]), 'the weighs are incorrect')
        #evaluate method validating:
        self.assertEqual(next(N.evaluate(ins)), oo, 'invalid output')

    def ftest_network_learning_1(self):
        W = [random() for x in range(12)]
        setup = 5, 2, 1
        ins = [1, 2, 3, 4, 5]
        N = Network(setup)
        bias = 0.05
        N[0][0][-1] = N[0][1][-1] = N[1][0][-1] = bias #biases are zeros
        N[0][0][:5] = W[:5]
        N[0][1][:5] = W[5:10]
        N[1][0][:2] = W[10:]
        #expected result:
        o1 = out(W[:5] + [bias], ins)
        o2 = out(W[5:10] + [bias], ins)
        oo = sigmoid(o1*W[10] + o2*W[11] + bias)
        t = 1
        delta_k = oo*(1 - oo)*(oo - t)
        psai_k = W[10]*delta_k, W[11]*delta_k
        W[10] -= o1*delta_k
        W[11] -= o2*delta_k

        #hidden layer weighs correction
        #first neuron:
        delta_o1 = o1*(1 - o1)*psai_k[0]
        for i in range(5):
            W[i]-=ins[i]*delta_o1
        delta_o2 = o2*(1 - o2)*psai_k[1]
        for i in range(5, 10):
            W[i]-=ins[i - 5]*delta_o2
        E = N.learn(ins, (t,))
        #error checking:
        self.assertEqual(E, 0.5*(oo - t)**2)
        #weighs correction from k-1 to k:
        self.assertEqual(N[1][0], 
                W[10:] + [-delta_k], 
                "invalid weight correction on the output layer")
        #validating the hidden layer neurons' weithgs:
        self.assertEqual(
                N[0][0],
                W[:5] + [-delta_o1], 
                "invalid weight correction on the hidden layer")
        self.assertEqual(N[0][1], 
                W[5:10] + [-delta_o2],
                "invalid weight correction on the hidden layer")
    
    def ftest_network_learning_2(self):
        setup = (6, 2, 3, 1)
        ins = [1, 99, 1, 11, 3, 123]
        t = 1
        
        N = Network(setup)
        W = []
        for i in range(0, len(setup) - 1):
            W.append([random() for x in range(setup[i]*setup[i + 1])])
            for j in range(setup[i + 1]):
                N[i][j][:setup[i]] = W[i][j*setup[i]:(j + 1)*setup[i]]

        #structure validating:
        test_i = randint(1, len(setup) - 1)
        self.assertEqual(len(N[test_i-1]), setup[test_i])
        self.assertEqual(len(N[test_i-1][0]) -1, setup[test_i-1])

        #forward iteration 1:
        Oi1 = out(W[0][0:6], ins)
        Oi2 = out(W[0][6:12], ins)
        i_outs = Oi1, Oi2
        
        #forward iteration 2:
        Oj1 = out(W[1][0:2], i_outs)
        Oj2 = out(W[1][2:4], i_outs)
        Oj3 = out(W[1][4:6], i_outs)
        j_outs = Oj1, Oj2, Oj3
        
        #forward iteration 3:
        Ok = out(W[2], j_outs)

        #backward iteration 1:
        delta_k = Ok*(1 - Ok)*(Ok - t)
        psai = (W[2][0]*delta_k, W[2][1]*delta_k, W[2][2]*delta_k)
        W[2][0] -= Oj1*delta_k
        W[2][1] -= Oj2*delta_k
        W[2][2] -= Oj3*delta_k

        #helping func:
        def calc_delta(O, P):
            return list(map(lambda o, p: o*(1 - o)*p, O, P))
        #backward iteration 2:
        delta_J = calc_delta(j_outs, psai)
        psai = ((
            W[1][0]*delta_J[0] + 
            W[1][2]*delta_J[1] + 
            W[1][4]*delta_J[2]
            ),
            W[1][1]*delta_J[0] + 
            W[1][3]*delta_J[1] + 
            W[1][5]*delta_J[2])

        W[1][0] -= Oi1*delta_J[0]
        W[1][1] -= Oi2*delta_J[0]

        W[1][2] -= Oi1*delta_J[1]
        W[1][3] -= Oi2*delta_J[1]
        
        W[1][4] -= Oi1*delta_J[2]
        W[1][5] -= Oi2*delta_J[2]

        #backward iteration 3:
        delta_I = calc_delta(i_outs, psai)
        for i in range(len(delta_I)):
            for j in range(6):
                W[0][i*6 + j] -= ins[j - i*6]*delta_I[i]

        E = N.learn(ins, (t,))
        self.assertEqual(E,0.5*(Ok-t)**2)
        #layer k checking:
        self.assertEqual(N[2][0], W[2] + [-delta_k])
        #layer j checking:
        self.assertEqual(N[1][0], W[1][0:2] + [-delta_J[0]])
        self.assertEqual(N[1][1], W[1][2:4] + [-delta_J[1]])
        self.assertEqual(N[1][2], W[1][4:6] + [-delta_J[2]])
        #leyar i checking:
        self.assertEqual(N[0][0], W[0][:6] + [-delta_I[0]])
        self.assertEqual(N[0][1], W[0][6:] + [-delta_I[1]])

class TestMutableNeuralNetwork(unittest.TestCase):
    
    def test_instance(self):
        layers = (5, 4, 1)
        m = MutableNetwork(*layers, n = 0.05)

        self.assertSequenceEqual(m.shape, layers)
        self.assertEqual(m.ninps, layers[0])
        self.assertEqual(m.nouts, layers[-1])
        self.assertEqual(m.speed, 0.05)
        self.assertEqual(repr(m), 
                "MutableNetwork({}, {}, {}, n = 0.05)".format(*layers))
        self.assertTrue(HiddenLayers(m)) #true when the network has at 
        #least one hidden layer, false otherwise
        self.assertTrue(InputLayer(m)) #always true
        self.assertTrue(OutputLayer(m)) #alwayes true
        self.assertFalse( #__len__() is called to consider bool value
                HiddenLayers(MutableNetwork(9, 2))) #len == 0 (no h-layers)

        self.assertIsNot(InputLayer(m), m.input_layer)
        self.assertIsNot(HiddenLayers(m), m.hidden_layers)
        self.assertIsNot(OutputLayer(m), m.output_layer)

        self.assertRaises(ValueError, MutableNetwork, 3) #too few args
        self.assertRaises(ValueError, MutableNetwork, 3, 0, 1) #empty layer
        self.assertRaises(ValueError, MutableNetwork, 8, -4, 1) #negative
        
        self.assertRaises(TypeError, MutableNetwork, 10, 9, 7, 1.01, 3)
        self.assertRaises(TypeError, MutableNetwork, 1.2, 9) #float size
        with self.assertRaises(TypeError):
            #verify that the speed coefficient cannot be non-numbers
            MutableNetwork(3, 2, n = 'str')
        self.assertRaises(TypeError, HiddenLayers, [0, 1])
        self.assertRaises(TypeError, OutputLayer, 99)
        self.assertRaises(TypeError, InputLayer, 'shrub')

    def test_primitives(self):
        self.assertIsInstance(Primitives.initweight(), _Number)
        #init sequence test:
        w_cnt = randrange(1, _SIZE)
        weights = Primitives.initweights(w_cnt)
        self.assertIsInstance(weights, _Sequence)
        for w in weights:
            self.assertIsInstance(w, _Number)
        self.assertEqual(len(weights), w_cnt)
        self.assertEqual(len(Primitives.initneuron(w_cnt)), w_cnt + 1)
        #init layer test:
        n_cnt = randrange(1, _SIZE)
        lyr = Primitives.initlayer(n_cnt, w_cnt)
        self.assertEqual(len(lyr), n_cnt)
        for n in lyr:
            self.assertIsInstance(n, _Sequence)
            self.assertEqual(len(n), w_cnt + 1)

        #adjusting layer:
        wshift = randrange(1, _SIZE)
        Primitives.adjustlayer(lyr, w_cnt + wshift)
        for n in lyr:
            self.assertIsInstance(n, _Sequence)
            self.assertEqual(len(n), w_cnt + wshift + 1)
            for w in n:
                self.assertIsInstance(w, _Number)

        self.assertRaises(ValueError, Primitives.initneuron, 0)
        self.assertRaises(TypeError, Primitives.initlayer, 2.1, 99)
        self.assertRaises(ValueError, Primitives.initlayer, 
                randrange(-_SIZE, 1), 99)
        self.assertRaises(ValueError, Primitives.initlayer, 3, 
                randrange(-_SIZE, 1))
        self.assertRaises(ValueError, Primitives.adjustlayer, lyr,
                randrange(-_SIZE, 1))

    def test_neuron_view_1(self):
        """Tests basic operations of NeuronView"""
        test_nlinks = randrange(1, _SIZE)
        test_seq = Primitives.initneuron(test_nlinks)
        test_view = NeuronView(test_seq)
        eval_view = eval(repr(test_view))

        self.assertEqual(eval_view, test_view)
        self.assertSequenceEqual(eval_view, test_view)
        
        self.assertEqual(eval_view, test_seq)
        self.assertSequenceEqual(eval_view, test_seq[:-1])

        self.assertEqual(len(test_view), test_nlinks)
        self.assertEqual(test_view.bias, test_seq[-1])
        self.assertSequenceEqual(test_view.weights, test_seq[:-1])
        self.assertSequenceEqual(test_view.tolist(), test_seq)
        
        #simple mutation:
        test_index = randrange(len(test_view))
        test_value = uniform(-_SIZE, _SIZE)
        test_view[test_index] = test_value
        self.assertEqual(test_view[test_index], test_value)
        self.assertEqual(test_view[test_index], test_seq[test_index])

        test_seq[test_index] = uniform(-_SIZE, _SIZE)
        self.assertEqual(test_view[test_index], test_seq[test_index])
        self.assertFalse(test_view.tolist() is test_seq) #if copy

        #errors:
        self.assertRaises(IndexError, test_view.__getitem__, 
                choice([-1, 1])*len(test_seq))
        self.assertRaises(TypeError, test_view.__getitem__, 2.33)
        self.assertRaises(TypeError, test_view.__setitem__, 
                "cowabunga dude!")
        with self.assertRaises(TypeError):
            test_view.bias = "Pokemon"

    def test_neuron_view_2(self):
        """Tests set and get methods of NeuronView by using mixed slices"""
        sz = _SIZE
        for x in range(_NUM):
            nlinks = randrange(1, sz)
            links = Primitives.initneuron(nlinks)
            view = NeuronView(links)

            tslice = slice(randrange(0, sz), randrange(0, sz), 
                    choice([1, -1])*randrange(1, 4))
            self.assertSequenceEqual(links[:-1][tslice], view[tslice])
            slice_len  = len(range(*tslice.indices(len(view))))
            new_weights = Primitives.initweights(slice_len)
            view[tslice] = new_weights
            
            self.assertSequenceEqual(links[:-1][tslice], new_weights)
            self.assertSequenceEqual(links[:-1][tslice], view[tslice])
            
            self.assertRaises(ValueError, view.__setitem__, tslice, 
                    [0]*(nlinks + 1))
            self.assertRaises(ValueError, view.__getitem__, 
                    slice(None, None, 0))
            self.assertRaises(ValueError, view.__setitem__, 
                    slice(None, None, 0), [])
            self.assertRaises(TypeError, view.__setitem__, 
                    slice(None, None), ['spam']*len(view))

            indx = randrange(len(view))
            with self.assertRaises(TypeError):
                view[randrange(len(view))] = 'S'

            self.assertEqual(view[indx:indx], [])
            
    def test_get_hlayer(self):
        for x in range(_NUM):
            M = TestMutableNeuralNetwork.__inititem(minsize = 3)
            L = M.hidden_layers
            test_start = randrange(0, len(L)) #target index
            test_slice = slice(test_start, randrange(test_start, len(L)))
            self.assertTrue(
                    all(all(isinstance(n, NeuronView) for n in lyr) 
                        for lyr in L[test_slice]), 
                    'inappropriate type of returning values')
            #check correspondence between M._links[s] and M[s]
            for Mi, Li in zip(M._links[test_slice], L[test_slice]):
                for Mij, Lij in zip(Mi, Li):
                    self.assertSequenceEqual(Mij[:-1], Lij) #only weights
                    self.assertSequenceEqual(
                            Mij[:-1], Lij.weights) #only weights
                    self.assertEqual(Mij[-1], Lij.bias) #only bias term
                    self.assertSequenceEqual(Mij, Lij.tolist())

            #check slices at random layer
            test_layer_index = test_start
            test_layer = tuple(L[test_layer_index])
            test_start = randrange(len(test_layer))
            test_slice = slice(test_start, 
                    randrange(0, len(test_layer)))
            self.assertTrue(all(
                isinstance(n, NeuronView) for n in test_layer[test_slice]),
                'inappropriate tupe of returning values')
            for m, n in zip(M._links[test_layer_index][test_slice], 
                    test_layer[test_slice]):
                self.assertSequenceEqual(m[:-1], n) #only weights
                self.assertSequenceEqual(m[:-1], n.weights) #only weights
                self.assertEqual(m[-1], n.bias) #only bias term
                self.assertSequenceEqual(m, n.tolist())
            
            #weird index combinations:
            self.assertEqual(L[test_layer_index:test_layer_index], tuple())
            self.assertEqual(list(L[test_layer_index, 
                test_start:test_start]), [])

            self.assertSequenceEqual(tuple(L[test_layer_index, :]), 
                    tuple(L[test_layer_index]))
            self.assertEqual(list(L[test_layer_index, 
                test_slice.start:len(test_layer):-1]), [])

            #exceptions:
            self.assertRaises(TypeError, L.__getitem__, (test_slice, 1)
                    #the double index notation should not accept slices
                    #at the first part
                    )
            self.assertRaises(IndexError, L.__getitem__, len(M._links) - 1
                    #the hidden layer view allow access by input layer's 
                    #index
                    )
            self.assertRaises(IndexError, L.__getitem__, 
                    choice([1,-1])*(randrange(0, _SIZE) + len(M._links)),
                    #index error is not raised when the given index is
                    #out of range
                    )
            self.assertRaises(ValueError, L.__getitem__, 
                    slice(test_slice.start, test_slice.stop, 0)
                    ) #slice step cannot be zero

    def test_insert_hlayer_1(self):
        """
        Verify inserting a new layer using simple sequences
        """
        for x in range(_NUM):
            M = TestMutableNeuralNetwork.__inititem()
            L = M.hidden_layers
            test_index = randint(0, len(L)) #target index
            #height of the current layer:
            height = M.shape[test_index] #number of inpurts per neuron
            test_layer = Primitives.initlayer(randrange(1, _SIZE), height)
            L.insert(test_index, test_layer)
            #next layer:
            next_layer = M._links[test_index + 1] #adjusted layer
            inserted_layer = M._links[test_index] #inserted layer
            #if the next layer consistent:
            self.assertTrue(
                    all(len(l) - 1 == len(test_layer) for l in next_layer),
                    "lost consistency after insertion")
            self.assertEqual(inserted_layer, test_layer, 
                "the inserted layer does not match the initial layer")
            #checking if there is no dependencies between the inserted 
            #layer and the source layer:
            self.assertTrue(all(not l is v for l, v in 
                zip(test_layer, inserted_layer)), 
                "inserted layer has a dependency on its source")
            
            fake_layer = Primitives.initlayer(randrange(1, _SIZE), height)
            fake_layer[randrange(len(fake_layer))].append(0) #contains the 
            #wrong number of weights
            
            #run with fake values:
            self.assertRaises(ValueError, L.insert, test_index, fake_layer)
            self.assertRaises(TypeError, L.insert, randint(0, len(L)), 
                    ["Em-bin-gun-do", "Endin-bo-go"])
            self.assertRaises(ValueError, L.insert, 
                    test_index, [[0]*height])

    def test_insert_hlayer_2(self):
        """
        Verify inserting a new layer using mixed sequences 
        """
        for x in range(_NUM):
            M = TestMutableNeuralNetwork.__inititem()
            L = M.hidden_layers
            test_index = randint(0, len(L)) #target index
            #height of the current layer:
            height = M.shape[test_index]
            test_layer = tuple(choice([tuple, NeuronView, list])(
                Primitives.initweights(height + 1)) for x in range(
                    randrange(2, _SIZE)))
            L.insert(test_index, test_layer)
            next_layer = M._links[test_index + 1] #adjusted layer
            self.assertTrue(all(len(l) - 1 == len(test_layer) 
                for l in next_layer), "lost consistency after insertion")
            
            self.assertRaises(ValueError, L.insert, test_index, 
                    [NeuronView(Primitives.initweights(height + 2)) 
                        for x in range(randrange(1, _SIZE))])
    
    def test_delete_hlayer(self):
        for x in range(_NUM):
            M = TestMutableNeuralNetwork.__inititem(minsize = 3)
            L = M.hidden_layers
            test_index = randrange(0, len(L)) #target index
            #delete one lyr:
            del L[test_index]
            links_per_neuron = M.shape[test_index]
            next_layer = M._links[test_index]
            self.assertTrue(all(len(n) - 1 == links_per_neuron 
                for n in next_layer), "lost consistency after deleting")
            #delete all:
            del L[:]
            self.assertTrue(all(len(n) == M.ninps #'n' is NeuronView 
                for n in M.output_layer), "lost consistency in the "
                "output layer after deleting all hidden layers")

    def test_setget_hlayer(self):
        """Tests assignment, deletion and evaluation implementations of 
        HiddenLayers (__setitem__, __delitem__, __getitem__)"""
        for x in range(_NUM):
            M = TestMutableNeuralNetwork.__inititem()
            #to ensure that the count of inputs does not changes:
            ninps = M.ninps #initial count of inputs
            H = M.hidden_layers
            shape = M.shape
            start = randrange(len(M) - 1)
            inset = [Primitives.initlayer(
                randrange(1, _SIZE), shape[start])]
            inset.append(Primitives.initlayer(randrange(1, _SIZE), 
                len(inset[0])))
            H[start:] = inset
            for o in M.output_layer:
                self.assertEqual(len(o), len(inset[1]))
            #deletion test:
            del H[:]
            for o in M.output_layer:
                self.assertEqual(len(o), ninps)

    def test_output_layer_1(self):
        """Tests basic operations of OutputLayer"""
        for x in range(_NUM):
            M = TestMutableNeuralNetwork.__inititem()
            O = M.output_layer
            rindex = randrange(len(O)) #random index
            view = O[rindex]
            tnode = M._links[-1][rindex] #target node
            self.assertSequenceEqual(view, tnode[:-1])
            self.assertEqual(view.bias, tnode[-1])
            #mutaion test:
            windex = randrange(len(view))
            wvalue = randrange(-_SIZE, _SIZE)
            view[windex] = wvalue

            self.assertEqual(tnode[windex], wvalue)
            #check how consistency is reestablished after mutation:
            links = M.shape[-2]
            if M.nlayr > 2:
                H = M.hidden_layers
                H[-1, links:] = [Primitives.initneuron(M.shape[-3])]
                b = len(tuple(H[-1, :]))
            else:
                I = M.input_layer
                I.insert(len(I))
            
            self.assertEqual(len(view), links + 1)
            self.assertEqual(len(tnode), links + 2)
            self.assertSequenceEqual(tnode, view)

    @staticmethod
    def __inititem(minsize = 2):
        return MutableNetwork(*(
            randrange(1, _SIZE) for x in range(randrange(minsize, _SIZE))))


###########################################################################
### Run tests
###########################################################################

def _test():
    unittest.main(verbosity = 2)

#run test:
if __name__ == '__main__':
    _test()
