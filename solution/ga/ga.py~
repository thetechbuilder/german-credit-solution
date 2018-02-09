#License: Public Domain
#
#ga.py
#

"""Base classes for expressing simple genetic algorithms 

Unit tests are in the current directory (test_ga.py, test.py)
"""

from multiprocessing import Pool, cpu_count
from random import uniform
from bisect import bisect
from abc import ABCMeta, abstractmethod
from collections import MutableSequence #already have ABCMeta

__all__ = ['GaIndividualBase', 'GaSeqBase','GaPopulationBase',
        'GaOperators', 'GaBase', 'MulticoreGaBase']

class GaIndividualBase(object):
    """
    GaIndividualBase can be used to test wheather a class provides 
    fitness value
    """
    def __init__(self):
        self.__fitness = float('-inf')

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, val):
        self.__fitness = val

class GaSeqBase(MutableSequence):
    """
    GaSeqBase provides basic methods for sequence structures such as set 
    of operators or set of individuals (population)

    Provides interface for random selection algorithm, so it can be used to
    test wheather a class provides random selection
    """
    def __getitem__(self, key):
        """
        G.__getitem__(y) <==> G[y]
        Returns specified individual
        """
        if isinstance(key, slice):
            return [self.get(i) for i in range(*key.indices(len(self)))]
        return self.__handle_index(key, self.get)
    
    def __setitem__(self, key, val):
        if isinstance(key, slice):
            for i, v in zip(range(*key.indices(len(self))), val):
                self.set(i, v)
        else:
            self.__handle_index(key, self.set, val)

    @abstractmethod
    def select(self):
        raise NotImplementedError()
    @abstractmethod
    def get(self, index):
        raise NotImplementedError()
    @abstractmethod
    def set(self, index, value):
        raise NotImplementedError()

#private methods:
    def __handle_index(self, index, func, *args):
        """
        Method for safe handling of indices
        """
        if isinstance(index, int):
            if index < 0: #handle negative indices
                index += len(self)
            if index > len(self) or index < 0:
                raise IndexError("Index {} is out of range".format(index))
            return func(index, *args)
        else:
            raise TypeError('Indices must be integers, not %s'%type(index))

class GaPopulationBase(GaSeqBase):
    """
    Provides basic infrastructure for population object without selection
    mechanism. Subclasses must implement specific selection mechanism.
    """
    def __init__(self, items, pressure = 10):
        """
        GaPopulation(pressure) -> population-object
        The 'pressure' argument describes tha amount discrimination of 
        selection mechanism. A system with a strong selection pressure very
        highly favours the more fit individuals, while system with a weak
        selection pressure very highly favours the more fit individuals
        """
        self._pressure = pressure
        self.items = [] if not items else list(items)

    def __len__(self):
        """
        P.__len__() <==> len(P)
        Returns the number of individuals
        """
        return len(self.items)

    def __delitem__(self, index):
        del self.items[index]

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        self._pressure = value

    def insert(self, index, value):
        self.items.insert(index, value)
    
    def get(self, index):
        return self.items[index]
    
    def set(self, index, val):
        self.items[index] = val

class GaOperators(GaSeqBase):
    def __init__(self, operators, weights):
        self.__operators = []
        self.__weights = []
        for val in zip(operators, weights):
            self.append(val)
    
    def __len__(self):
        """
        G.__len__() <==> len(G)
        """
        return len(self.__operators)

    def __delitem__(self, i):
        """
        G.__delitem__(o) <==> del G[o]
        """
        del self.__operators[i]
        W_bias = self.__getw(i)
        del self.__weights[i]
        self.__shiftw(i, W_bias)

    def get(self, index):
        """
        G.get(i) -> (operator, weight)
        Returns the operator and its weight at the specified index i
        
        *this method does not contain any validation of input values
        """
        return self.__operators[index], self.__getw(index)
    
    def set(self, index, value):
        """
        G.set(index, y) -- set operator y at the specified index,
        where the argument y consists of operator and its weight
        
        *this method does not contain any validation of input values
        """
        self.__operators[index] = value[0]
        self.__shiftw(index, self.__getw(index) - value[1])

#public methods:
    def insert(self, index, value):
        """
        G.append(val) -- append an operator to the end
        """
        o, w = value
        try:
            is_neg = w < 0
        except TypeError:
            raise ValueError('Weight of the operator '
                    '{} is not a number ({})'.format(type(o), w))
        if is_neg:
            raise ValueError('Weight of the operator '
                    '{} is negative ({})'.format(type(o), w))
        if w > 0 and not o in self.__operators:
            self.__operators.insert(index, o)
            self.__weights.insert(index,
                    self.__weights[-1] + w if len(self.__weights) else w)
            self.__shiftw(index + 1, w) #shift values

    def select(self):
        """
        G.select() -> function
        Returns the function from the specified set of operators selected 
        at random according to the specified weights (proportionate)
        """
        return self.__operators[bisect(self.__weights, 
            uniform(0, self.__weights[-1]))]
    
    def index(self, o):
        """
        G.index(operator) -> integer
        Returns the index of the specified operator, None otherwise
        """
        for i in range(len(self)):
            if self.___operators[i] == o:
                break
        else:
            i = None
        return i

    def weight(self, o, d = None):
        """
        G.get(o [,d]) -> G[o] if o in G, else d. d defaults to None
        """
        i = self.index(o)
        return d if i == None else self.__getw[i]

#private methods:
    def __getw(self, W_index):
        """
        Returns the weight of 
        """
        W = self.__weights
        return W[W_index] - W[W_index - 1] if W_index else W[W_index]

    def __shiftw(self, W_index, W_bias):
        """
        Shifts weights
        """
        W = self.__weights
        for i in range(W_index, len(W)):
            W[i] -= W_bias

class GaBase(metaclass = ABCMeta):
    def __init__(self, population, operators):
        if not isinstance(population, GaSeqBase):
            raise TypeError('Initial population must'
                    'be inherited by GaPopBase')
        if not isinstance(operators, GaOperators):
            raise TypeError('Genetic operators object must be derived '
                    'from GaPopulation class')
        self.population = population
        self.operators = operators

    @abstractmethod
    def evolve(self):
        raise NotImplementedError()

class MulticoreGaBase(GaBase):
    def __init__(self, population, operators):
        GaBase.__init__(self, population, operators)
        self._pool = Pool(processes = cpu_count())

    def execute_operator(self):
        return self.operators.select()(self.population)

    def evolve(self):
        r = range(len(self.population))
        #self._pool.map(self.execute_individual, r)
        #self.population[:] = self._pool.map(self.execute_operator, r)
        for i in r:
            self.execute_individual(i)
        l = [self.execute_operator() for i in r]
        self.population[:] = l

    @abstractmethod
    def execute_individual(self, i):
        raise NotImplementedError()
