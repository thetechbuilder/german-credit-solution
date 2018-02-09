#License: Public Domain
#
#neuro.py
#
"""Implements fully-connected forward-feed neural network model
with unconstrained dimmensions.

Unit tests are in the current directory (test_neuro.py).
"""
#standard modules:
import math
from copy import deepcopy
from random import uniform
from itertools import chain
from abc import abstractmethod
from abc import ABCMeta as _ABCMeta
from numbers import Number as _Number
from collections.abc import Sized as _Sized, Iterable as _Iterable
from collections.abc import Sequence as _Sequence
from collections.abc import MutableSequence as _MutableSequence

__all__ = ["TrainingAlgorithmBase", "Primitives", "Network", 
        "Algorithms", "NeuronView", "InputLayer", 
        "OutputLayer", "HiddenLayers", "MutableNetwork"]

#ABSTRACT CLASSES
class TrainingAlgorithmBase(metaclass = _ABCMeta):
    """
    The abstract base class for all training algorithms
    """
    #Some algorithms might be implemented as objects in order to improve
    #some parameters. For example, as an object, training algorithm can 
    #hold some parameters of a network structure when learing. 
    #Consequently, I have decided to implement training algorithms 
    #as objects so that it is possible to nest additional parameters.
    def __init__(self, network):
        if not isinstance(network, Network):
            raise TypeError("Unacceptible type of the specified object. "
                    "'network' must be either an instance of Network "
                    "or a subclass thereof, not %s"%type(network))
        self._nn = network
    
    def __str__(self):
        """
        The “informal” string representation of an object for print 
        statements
        """
        return ("<Training algorithm: "
                "{} emeded to the network at {:x}>".format(
                    self.__class__.__name__, id(self)))

    @abstractmethod
    def learn(self, selection, target):
        """
        Describes the method of learning
        """
        raise NotImplementedError()

    @abstractmethod
    def activate(self, ins):
        """
        Describes the method of activation
        """
        raise NotImplementedError()

#HElPING CLASSES
class Primitives:
    """
    Implements the building blocks for constructing MLP-structures.
    These methods are protected from invalid values.

    On account of the fact that realatively big algorithms can be very
    susceptible to the case of running to a completion and producing 
    a wrong answer that could be hard to diagnose or eliminate, I have
    wrote some helping method improveing the rigor of the semantics 
    so that it is possible to weed some things out.
    """
    @staticmethod
    def initweight():
        """
        Initializes the weight generated according to the accepted pattern.
        """
        return uniform(-0.05, 0.05)
    
    @staticmethod
    def initweights(wcount):
        """
        Returns the list of random numbers in the range of [-0.05, 0.05], 
        which is used for initialization of weights when adding new 
        neurons to networ structures.
        """
        return [Primitives.initweight() for i in range(wcount)]

    @staticmethod
    def initneuron(nlinks):
        """
        Primitives.initneuron(nlinks) <==> initweights(nlinks + 1)
        """
        if nlinks < 1:
            raise ValueError("the count of links cannot be less than one, "
                    "{} given".format(nlinks))
        return Primitives.initweights(nlinks + 1)

    @staticmethod
    def initlayer(ncount, nlinks):
        """
        Returns the layer object generated according to the specified 
        paramethers
        """
        if not isinstance(ncount, int):
            raise TypeError("The size of the layer must be an integer, "
                    "not {}".format(type(ncount)))
        if ncount < 1:
            raise ValueError("the count of neurons cannot be less than one"
                    ", {} given".format(ncount))
        return [Primitives.initneuron(nlinks) for i in range(ncount)]

    #mutation functions:
    @staticmethod
    def adjustlayer(layer, wcount):
        wcount += 1 #actual count of weights
        for n in layer:
            Primitives.adjustnode(n, wcount)
        return layer

    @staticmethod
    def adjustnode(node, wcount):
        if wcount < 2:
            raise ValueError("the count of links cannot be less than one, "
                    "{} given".format(wcount))
        if len(node) > wcount:
            del node[0:len(node) - wcount]
        else:
            node[-1:-1] = Primitives.initweights(wcount - len(node))
        return node

    @staticmethod
    def copy(c):
        return deepcopy(c)

#MAIN CLASSES:
class Algorithms:
    class SigmoidBackPropagation(TrainingAlgorithmBase):
        """
        Implements the back propagation method of training multilayer 
        perceptron networks.
        """
        def __init__(self, network):
            TrainingAlgorithmBase.__init__(self, network)
            #Initialized and reestablishes list of outputs:
            self.__outputs = tuple([0]*ln for ln in self._nn.shape)
        
        #private methods:
        @staticmethod
        def __get_outputs(O, Wl):
            """
            __get_outputs(O, Wl) -> map obgect
            Returns the next layer outputs
            Arguments:
            1) O is an outputs of some layer n
            2) Wl is weights from some layer n to n + 1
            """
            return map(lambda W: Network.__sigmoid(
                sum(map(lambda o, w: o*w, O, W[:-1])) + W[-1]), Wl)
    
        @staticmethod
        def __sigmoid(u): #activation function
            """
            Represents activation function of Uni-polar sigmoid function.
            It produces output values in the range of [0, 1]. It is also
            known as binary sigmoid function.
            """
            return 1/(1 + math.e**-u)
    
        @staticmethod
        def __psai(layer, delta):
            """
            Returns the result of multiplication of layer and transposed 
            delta
            """
            #len(layer[0]) - 1 because the last argument is the bias
            return (sum(delta[i]*layer[i][j] for i in range(len(delta))) 
                    for j in range(len(layer[0]) - 1))
    
        @staticmethod
        def __error(o, t):
            """
            Returns the error
            """
            return 0.5*(o - t)**2
        
        def __update_layer(self, i, delta):
            """
            Updates weights of the neurons from i - 1 to i layer
            
            Arguments:
            1) 'i' is the index of layer
            2) 'delta' is the constant part of the derivative of 
            the error with
            respect to weight from i - 1 to i
            3) 'n' is the learning speed
            """
            W = self._nn._links[i] #weights from i-1 to i layer
            O = self.__outputs[i]  #go to the previous
            for j in range(len(W)): #for each neuron index
                #j is the neuron index
                dw = self.__n*delta[j] #reqular part for the next loop
                #delta depends on neuron
                for k in range(len(O)): #for each output
                    #k is the weight index
                    W[j][k] -= O[k]*dw #weight correction
                W[j][-1] -= dw #bias term correction

        def __forward(self, ins):
            """
            N.__forward_prop(*ins)
            Forward propagation
            """
            self.__outputs[0][:] = ins #initial output
            #range(len(self.__outputs) - 1) == range(len(self.__inputs))
            for i in range(len(self.__outputs) - 1): #starts from hidden 
                #layers
                Oi = self.__outputs[i] #outputs from nodes from i-th layer
                #wgs is the weighs from i to j, i.
                Wl = self._nn._links[i] #all weighs from i to i + 1 layer
                self.__outputs[i + 1][:] = Network.__get_outputs(Oi, Wl)
            return self.__outputs[-1]
        
        def __backward(self, t):
            """
            Realizes the backward propagation algorithm under the current 
            neural netwok, i.e. corrects the weights
            """
            W = self._nn._links #weights
            O = self.__outputs
            #ouput layer processing:
            delta = tuple(map(
                lambda o, t: o*(1 - o)*(o - t),
                O[-1], t)) #The output layer outputs
            psi = list(Network.__psai(W[-1], delta)) #transition delta
            self.__update_layer(len(W) - 1, delta) #update last layer
            #hidden layers processong:
            for i in range(len(W) - 2, -1, -1):
                #i is a layer index
                delta = tuple(map(lambda o, p: o*(1 - o)*p, O[i + 1], psi))
                psi[:] = Network.__psai(W[i], delta)
                self.__update_layer(i, delta)

        def learn(self, selection, target):
            outs = self.__forward(selection)
            self.__backward(target)
            return sum(map(Network.__error, outs, target))

        def activate(self, ins):
            """
            N.evaluate(ins) -> map object -- output of neural network
            """
            W = self._nn._links
            for lr in range(len(W)):
                ins = Network.__get_outputs(ins, W[lr])
            return ins

class Network(object):
    """
    Network(layers) -> new empty artificial neural network
    Represents a multilayer perceptron (MLP), i.e the artificial neural 
    network model with back-propagation training algorithm.
    
    Properties:
    * Forward-feed
    * Fully connected
    """ 
    def __init__(self, *layers, n = 1, 
            algorithm = Algorithms.SigmoidBackPropagation):
        """
        x.__init__(layers) -> Network -- initialize x
        """
        
        if all(isinstance(i, _Number) for i in layers):
            if len(layers) < 2:
                raise ValueError("The neural network must contain at "
                        "least two layers, {} given".format(len(layers)))
            #generate structure accoring to the specified shape:
            self._links = list(
                    Primitives.initlayer(layers[n], layers[n - 1])
                    for n in range(1, len(layers)))
        elif all(isinstance(i, _Sequence) for i in layers):
            nlen = len(layers[0][0]) - 1 #desired count of links 
            #running into
            #into each neuron
            self._links = [0]*len(layers)
            for i, l in enumerate(layers):
                self._links[i] = [
                        _LinkedLayer.validate_individual(n, i, nlen) 
                        for n in l]
                nlen = len(l)
        else:
            raise ValueError("Consistency error. The specified layer "
                "structure cannot be accepted for proper initialization")
            
        self.speed = n
        self._ninps = len(self._links[0][0]) - 1
        self.algorithm = algorithm

    def __repr__(self):
        """N.__repr__() <==> repr(N)"""
        return "{}({}, n = {})".format(self.__class__.__name__,
                ", ".join(map(str, self.shape)), self.speed)

    def __len__(self):
        """
        Returns the current number of layers
        """
        return self.nlayr  #input layer is included

    def activate(self, ins):
        """
        N.evaluate(ins) -> map object -- output of neural network
        """
        return self.__algorithm.activate(ins)

    def learn(self, selection, target):
        """
        N.learn(*args)
        Argument:
        1) 'selection' is a lerning selection (sequence)
        2) 'target' is the target value of input nodes (sequence)
        """
        if len(selection) != self.ninps:
            raise ValueError(
                    'The input data has the wrong quantity of elements.'
                    'The given number of inputs is '
                    '{}, but {} is required.'.format(
                        len(selection), self.ninps))
        if isinstance(target, _Number):
            target = target,
        return self.__algorithm.learn(selection, target)

    #properties:
    @property
    def algorithm(self):
        """
        N.algorithm -> class -- current learing algorithm
        """
        return self.__algorithm.__class__

    @algorithm.setter
    def algorithm(self, val):
        if not issubclass(val, TrainingAlgorithmBase):
            raise TypeError("Unacceptable training algorithm. Training "
                    "algorithm must be a subclass of TrainingAlgorithmBase"
                    ", not {}".format(val))
        self.__algorithm = val(self)

    @property
    def speed(self):
        """
        N.speed -> int -- current speed
        """
        return self.__n

    @speed.setter
    def speed(self, val):
        try:
            self.__n = float(val)
        except:
            raise TypeError("The speed must be numerical, "
                    "not {}".format(type(val)))

    @property
    def shape(self):
        """
        N.shape -> tuple -- network dimensions
        Returns tuple of network dimensions. This tuple of integers 
        indicating the number of neurons in each layer including inputs.
        For neural network that has four inputs, three hidden and a single
        output neuron, the shape will be (4, 3, 1).
        """
        #concatenate the number of inputs and the series of the next layer
        #sizes:
        return (self.ninps, ) + tuple(len(self._links[i]) 
                for i in range(len(self) - 1))

    @property
    def size(self):
        """
        N.size -> int -- the number of neurons
        Returns the total number of neurons of the neural network. 
        This is equal to the product of the elements of shape.
        """
        return self.ninps + sum(len(l) for l in self._links)

    @property
    def nlayr(self):
        """
        N.nlayr -> int -- the number of actual layers
        Returns the number of layers, i.e the length of the shape.
        """
        return len(self._links) + 1
    
    @property
    def ninps(self):
        """
        N.inps -> int -- the number of inputs
        Returns the required number of inputs for activating.
        """
        return self._ninps

    @property
    def nouts(self):
        """
        N.outs -> int -- the number of outputs
        Returns the nuber of output neurons of the network.
        """
        return len(self._links[-1])


#MUTABLE NEURAL NETWORK CLASSES:
class _LayerView:
    def __init__(self, nn):
        if not isinstance(nn, Network):
            raise TypeError("Initialization value must be an instance of "
                    "Network, or of a subclass thereof. Not %s" %type(nn))
        self._nn = nn

    def __repr__(self):
        return "{}({})".format(self.__class__, repr(self._nn))

    def _refresh(self):
        """
        Designed only for internal use.
        This method is used for reloading algorithm after mutation.
        """
        a = self._nn.algorithm
        self._nn.algorithm = a

class _LinkedLayer(_LayerView):
    def _pllen(self, index):
        """
        Returns the number of links
        """
        return len(self._nn._links[index - 1]) if index else self._nn.ninps

    @staticmethod
    def validate_individual(v, layer_index, link_count):
        """
        Returns a valid and checked copy of the neuron (a bunch of weights 
        and bias specifically packed in the list), throws an error 
        otherwise.
        """
        if isinstance(v, NeuronView):
            if len(v) == link_count:
                return v.tolist()
            raise ValueError("The specified neuron "
                "{} does not match the layer {}".format(v, layer_index))
        elif not all(isinstance(i, _Number) for i in v):
            raise TypeError("The initial neuron parameters "
                "{} must be numbers".format(v))
        elif not len(v) == link_count + 1: #count of links + place for bias
            raise ValueError("The specified initial sequence "
                    "{} doesn't match the layer {}".format(v, layer_index))
        return list(v)

class NeuronView(_Sequence):
    def __init__(self, neuron):
        if not isinstance(neuron, _MutableSequence):
            raise TypeError(
                    "Initialization value must be a mutable sequence, "
                    "the specified %s is not"%type(neuron))
        self._neuron = neuron

    def __repr__(self):
        """n.__repr__() <==> repr(n)"""
        return "NeuronView({})".format(self._neuron)

    def __str__(self):
        """n.__str__() <==> str(n)"""
        return "<{}; {}>".format(", ".join(map(str, self._neuron[:-1])), 
                self._neuron[-1])

    def __len__(self):
        """n.__len__() <==> len(n) -- number of inputs (weights)"""
        return len(self._neuron) - 1 #bias term does not include

    def __eq__(self, l):
        """n.__eq__(l) <==> n == l"""
        if isinstance(l, NeuronView):
            return l.tolist() == self._neuron
        return l == self._neuron

    def __getitem__(self, key):
        """n.__getitem__(x) <==> n[x]"""
        return self._neuron[slice(*key.indices(len(self))) if 
                isinstance(key, slice) else _handleindex(key, len(self))]

    def __setitem__(self, key, value):
        """n.__setitem__(x, v) <==> n[x]"""
        if isinstance(key, slice):
            indices = key.indices(len(self))
            #ensure that the count of elements for copying is equal to 
            #the range of the specified slice
            if len(value) != len(range(*indices)):
                raise ValueError("The assigned values do not match the "
                        "range of the specified slice, i.e. "
                        "len(indices({})) != len(values({}))".format(
                            list(range(*indices)), value))
            elif not all(isinstance(v, _Number) for v in value):
                raise TypeError("The assined values must be numbers, i.e. "
                        "the values must be instances of numbers.Number")
            key = slice(*indices)
        else:
            key = _handleindex(key, len(self))
            if not isinstance(value, _Number):
                raise TypeError("The weights of the neuron must be numbers"
                        ", not {}".format(type(value)))
        self._neuron[key] = value #the arguments are trustad

    def tolist(self):
        """
        n.tolist() -> list
        Returns the list that contain all weights and bias that is used to 
        initialize this object, so n is equal NeuronView(n.tolist())
        """
        return self._neuron[:] #return copy of self._neuron

    @property
    def weights(self):
        return tuple(self._neuron[:-1])

    @weights.setter
    def weights(self, values):
        self[:] = values

    @property
    def bias(self):
        return self._neuron[-1]

    @bias.setter
    def bias(self, value):
        if isinstance(value, _Number):
            self._neuron[-1] = value
        else:
            raise TypeError("The specified value must be a number, "
                    "not {}".format(type(value)))

    @staticmethod
    def merge(weights, bias):
        """
        NeuronView.merge(weight, bias) -> merged representation

        Parses and verifies the specified sequence of weights with
        specified bias value into the merged internal representation.
        """
        weights = list(weights)
        if not isinstance(bias, _Number):
            raise TypeError("bias must be a number, not %s"%type(bias))
        elif not all(isinstance(w, _Number) for w in weights):
            raise TypeError("Weights must be numbers")
        elif len(weights) < 1:
            raise ValueError("The count of weights must be more than zero")
        return weights + [bias]

class InputLayer(_LayerView, _Sized):
    def __len__(self):
        """I.__len__() <==> len(I) -- the number of inputs"""
        return self._nn.ninps

    def __delitem__(self, key):
        """I.__delitem__(k) <==> del I[k] -- delete one input"""
        key = _handleindex(key, len(self))
        if len(self) < 2:
            raise ValueError("The input layer cannot be removed")
        #traverse the next layer and remove corresponding links:
        for x in self._nn._links[0]:
            del x[key]
        self._nn._ninps -= 1 #correct constant
        self._refresh()

    def insert(self, index):
        """
        Inserts input neuron before the index
        """
        index = _handleindex(index, self._nn.ninps + 1)
        #traverse the next layer and insert corresponding weights
        for s in self._nn._links[0]:
            s.insert(index, Primitives.initweight())
        self._nn._ninps += 1 #correct constant
        self._refresh()

    def append(self):
        self.insert(-1)


class OutputLayer(_LinkedLayer, _Sequence):
    def __len__(self):
        """O.__len__() <==> len(O) -- the number of output neurons"""
        return self._nn.nouts

    def __getitem__(self, key):
        """O.__getitem__(x) <==> O[x]"""
        if isinstance(key, slice):
            key = slice(*key.indices(len(self)))
            return map(NeuronView, self._nn._links[-1][
                slice(*key.indices(len(self)))])
        return NeuronView(
                self._nn._links[-1][_handleindex(key, len(self))])

    def __setitem__(self, key, value):
        """O.__getitem__(x, v) <==> O[x] = v"""
        L = self._nn._links
        nlinks = self._pllen(len(L) - 1) #number of links per neuron
        if isinstance(key, slice):
            key = key.indices(len(self))
            if not len(value) and len(range(*key)) == len(self):
                raise IndexError("The output layer cannot be removed")
            L[-1][slice(*key)] = (_LinkedLayer.validate_individual(
                val, -1, nlinks) for val in value)
        else:
            L[-1][_handleindex(key, len(self))
                    ] = _LinkedLayer.validate_individual(value, -1, nouts)
        self._refresh()

class HiddenLayers(_LinkedLayer, _MutableSequence):
    def __len__(self):
        return self._nn.nlayr - 2 #count of hidden neurons

    def __getitem__(self, key):
        """
        H.__getitem__(x) <==> H[x]

        This implementation accepts double-indexation, i.e using tuple of
        two indices to reach out the neuron.
        
        Example: H.__getitem__((l, n)) <==> H[(l, n)] <==> H[l, n] 
        - the argument 'l' (integer) indicates the layer;
        - the argument 'n'(integer of slice) indicates the neurons at 
        the l-th layer.    
        H[l] -> map of neurons if the argument 'l' is an integer or tuple 
        of layers if the argument 'l' is a slice.
        H[l, n]  -> NeuronView if 'n' is integer, map of neurons otherwise
        """
        L = self._nn._links
        if isinstance(key, _Sequence) and len(key) == 2:
            layer_index = _handleindex(key[0], len(self))
            neuron_index = key[1]
            if isinstance(neuron_index, slice):
                neuron_index = slice(*neuron_index.indices(
                    len(L[layer_index])))
                return map(NeuronView, L[layer_index][neuron_index])
            neuron_index = _handleindex(neuron_index, len(L[layer_index]))
            return NeuronView(L[layer_index][neuron_index])
        elif isinstance(key, slice):
            return tuple(map(NeuronView, l) for l in 
                L[slice(*key.indices(len(self)))]) 
        return map(NeuronView, L[_handleindex(key, len(self))])

    def __setitem__(self, key, value):
        #inappropriate values will be refused
        L = self._nn._links
        if isinstance(key, _Sequence) and len(key) == 2:
            layer_index = _handleindex(key[0], len(self))
            neuron_index = key[1]
            if isinstance(neuron_index, slice):
                neuron_index = slice(
                    *neuron_index.indices(len(L[layer_index])))
                old_layer_length = len(L[layer_index]) #initial length
                #throws ValueError if a step of the slice != 1 and 
                #len(value) != len(L(slice))
                L[layer_index][neuron_index] = (
                    _LinkedLayer.validate_individual(v, layer_index, 
                        self._pllen(layer_index)) for v in value)

                new_layer_length = len(L[layer_index])
                if not new_layer_length:
                    #the layer will be deleted if the count of neurons
                    #equals zero
                    del self[layer_index]
                elif old_layer_length != new_layer_length:
                    #adjust the next layer manually if the count of 
                    #neurons has been changed
                    
                    #Considering that the count of neurons can be lesser 
                    #or higher than the initial length after setting,
                    #the negative and positive mutation is handling
                    #separatly:
                    diff = new_layer_length - old_layer_length
                    stop = neuron_index.stop #last value
                    #In this case either start <= stop or (stop == 0 and 
                    #start >= stop).
                    if diff > 0:
                        for n in L[layer_index + 1]:
                            #inserts before n[stop]
                            n[stop:stop] = (Primitives.initweight() 
                                    for i in range(diff))
                    else:
                        #adjust the next layer, i.e remove all redundant 
                        #links
                        for n in L[layer_index + 1]:
                            #notice that the diff is negative
                            del n[stop + diff:stop]
            else:
                neuron_index = _handleindex(neuron_index, 
                    len(L[layer_index]))
                L[layer_index][ #one to one substitution
                    neuron_index] = _LinkedLayer.validate_individual(
                        value, layer_index, self._pllen(layer_index))
                #no adjusting
        elif isinstance(key, slice):
            #raises ValueError if slice step is lesser than 1:
            key = slice(*key.indices(len(self)))
            #with step or not
            if abs(key.step) > 1: #avoids the 1 and -1 step values
                #raises an error if there is attempt to assign 
                #a sequence of the specified size to an extended slice
                #of the unmatched size
                L[key] = ([_LinkedLayer.validate_individual(node, 
                    lr_i, self._pllen(lyr_i)) for node in value[lr_i]]
                    for lr_i in range(key.start, key.stop, key.step))
                #adjust last all front neighboors:
                for i in range(key.start + 1, key.stop + 1, key.step):
                    Primitives.adjustlayer(L[i], self._pllen(i))
            else:
                #no errors if sequences' lengths do not match
                #can also remove layer when using notation N[x] = []
                top_bound = key.start + len(value)
                L[key] = (
                    [_LinkedLayer.validate_individual(
                        node, nlyr, nlinks) for node in value[indx]]
                    for indx, nlinks, nlyr in zip(range(len(value)),
                        chain((self._pllen(key.start),), map(len, value)),
                        range(key.start, top_bound)))
                #adjust the layer in front of the slice:
                Primitives.adjustlayer(
                    L[top_bound], self._pllen(top_bound))
        else:
            key = _handleindex(key, len(self))
            L[key] = (_LinkedLayer.validate_individual(
                node, key, self._pllen(key)) for node in value)
            #adjust the next layer
            Primitives.adjustlayer(L[key + 1], len(L[key]))
        self._refresh() #reestablish learning algorithm

    def __delitem__(self, key):
        """H.__delitem__(k) <==> del H[k]"""
        if isinstance(key, _Sequence):
            self[key[0], key[1]: key[1] + 1] = ()
        elif isinstance(key, slice):
            self[key] = ()
        else:
            self[key:key + 1] = ()

    def insert(self, index, value):
        self[index:index] = (value, )

#private methods: 
    @staticmethod
    def __getitem(collection, key, length):
        if isinstance(key, slice):
            index = slice(*key.indices(length))
        else:
            index = _handleindex(key, nlayrs)
        return 
        if isinstance(key, _Sequence):
            return Network.__getitem(collection[int(key[0])], 
                    key[1:] if len(key) > 2 else key[1])
        return collection[key]

class MutableNetwork(Network):
    """
    MutableNetwork(layers) -> new empty mutable neural network

    Broadens the neural network implementation by adding additional 
    operations that allow modification of the network while saving 
    consistency of the network structure.

    On account of the fact that the base class implements fully connected 
    artificial neural network, in some cases you have to specify 
    corrective weights that is used to reestablish consistensy in the 
    network structure. After each neuron insertion, corrective weights 
    defined at uniform random inside the interval [-0.05, 0.05].
    """
    @property
    def input_layer(self):
        return InputLayer(self)

    @property
    def hidden_layers(self):
        return HiddenLayers(self)
    
    @property
    def output_layer(self):
        return OutputLayer(self)

#In many implementations, sequence-wise, handling of indices runs by 
#the same way; consequently, I wrote a snippet of the high-usage 
#code for validation:
def _handleindex(index, length):
    if isinstance(index, int):
        if index < 0: #handle negative indices
            index += length
        if index < 0 or not index < length:
            raise IndexError("The specified index "
                    "{} is out of range [{},{}]".format(index, 0, length))
    else:
        raise TypeError(
                'invalid type of the specified index, indices must be '
                'integers, not {} ({} given)'.format(type(index), index))
    return index
