#License: Public Domain
#
#gensyn.py
#
"""Genetic algorithm implementation for neural network synthesis

Unit tests are in the current directory (test_gensyn.py, test.py)
"""
#standard modules:
from random import sample, choice, randint, gauss, randrange
from itertools import chain
import math
#making 'ga' and 'nn' folders to be accessible
from sys import path
path.append('ga') #genetic algorithm implementation
path.append('nn') #neural network implementation
#included modules:
from ga import GaIndividualBase, GaPopulationBase, MulticoreGaBase #ABSs
from ga import GaOperators
from neuro import MutableNetwork, Primitives
from german_credit import net_profit

def update_progress(progress):
    print('\r [{0}] {1}%'.format('#'*int((progress/10)), 
        progress), end='')

class Individual(GaIndividualBase, MutableNetwork):
    def __init__(self, layers, n = 0.1, periods = 100000):
        GaIndividualBase.__init__(self)
        MutableNetwork.__init__(self, *layers, n = n)
        self.__periods = periods
        self.__prog = 0 #indicates evalutation progress

    @property
    def periods(self):
        return self.__periods
    
    @periods.setter
    def periods(self, value):
        self.__periods = value

    @property
    def current_period(self):
        """
        Return current lerning period, i.e a number of evaluation periods
        """
        return self.__prog

    def execute(self, training_set, validation_set):
        self.__prog = 0
        while self.__prog < self.periods:
            self.__prog += 1
            t = choice(training_set)
            error = self.learn(*t)
            update_progress(round(self.__prog/self.periods*100))
        update_progress(100)
        
        #specific for german credit valuse:
        vals = tuple(map(lambda x: next(self.activate(x)), 
                (v[0] for v in validation_set)))
        bound = (max(vals) + min(vals))/2
        net = fn = fp = 0
        for v, t in zip(vals, [vd[1] for vd in validation_set]):
            v = 1 if v > bound else 0
            if t == 0 and v == 1:
                #false negative classification
                fn += 1
            if t == 1 and v == 0:
                #false positive classification
                fp += 1
            net += net_profit(v, t)
        self.fitness = net, fp, fn
        return self.fitness

class Population(GaPopulationBase):
    def select(self):
        """
        P.choose() -> individual
        Selects individual by using tournament method
        
        In tournament selection a number of individuals are chosen 
        at random from the population. These are compared with each 
        other and the best of them is chosen to be the parent.
        * In this implementation selection pressure defines a number of
        individuals
        * Indivuduals are compared by "fitness" value of indivudual
        """
        #sample cannot be larger than population
        l = sample(range(len(self)), 
                self.pressure if self.pressure < len(self) else len(self))
        #select the best individual from the sample
        maxf, val = float('-inf'), None #max fitness value (maxf)
        for s in l:
            actual_fitness = (self[s].fitness[0] - self[s].size*2 - 
                        self[s].fitness[1]*10 - self[s].fitness[2]*10)
            if maxf < actual_fitness: #using net-profit solely
                val, maxf = self[s], actual_fitness
        return val

class Gensyn(MulticoreGaBase):
    def __init__(self, population, operators, 
            training_set, validation_set):
        MulticoreGaBase.__init__(self, population, operators)
        self.training_set = training_set
        self.validation_set = validation_set

    def execute_individual(self, i):
        self.population[i].execute(self.training_set, self.validation_set)
        print(" :: individual %s has been executed"%i)

class RawOperators:
    @staticmethod
    def crossover_1(population):
        P1, P2 = Primitives.copy(population.select()), population.select()
        H1_TARGET = P1.hidden_layers
        H2 = P2.hidden_layers
        if len(P1) == len(P2):
            if randint(0, 3): #is cross
                for i in range(len(H1_TARGET)):
                    h1_layer = tuple(H1_TARGET[i])
                    h2_layer = tuple(H2[i])
                    H1_TARGET[i,:] = chain(sample(h1_layer, 
                        math.ceil(len(h1_layer)/2)),
                        Primitives.adjustlayer(sample(tuple(h2.tolist() 
                            for h2 in h2_layer), math.ceil(
                                len(h2_layer)/2)), P1.shape[i]))
        else:
            i = 0
            for (h1, h2) in zip(H1_TARGET[:], H2[:]):
                if randint(0, 1):
                    H1_TARGET[i:i+1] = [Primitives.adjustlayer(
                            [x.tolist() for x in h2], P1.shape[i])]
                i+=1
            if len(P2) > len(P1):
                MAX_H = H2
            else:
                MAX_H = H1_TARGET
            inset = []
            for k in range(i, len(MAX_H)):
                if randint(0, 2):
                    inset.append([x.tolist() for x in MAX_H[k]])
            for l in range(1, len(inset)):
                Primitives.adjustlayer(inset[l], len(inset[l-1]))
            if inset:
                Primitives.adjustlayer(inset[0], P1.shape[i])
            H1_TARGET[i:] = inset
        return P1


    @staticmethod
    def reproduction(population):
        return Primitives.copy(population.select())

    @staticmethod
    def add_and_remove_neuron(population):
        P = Primitives.copy(population.select())
        H = P.hidden_layers
        for i in range(len(H)):
            index = randrange(P.shape[i + 1])
            if P.shape[i + 1] < 2 or randint(0, 1):
                new_neuron = Primitives.initneuron(P.shape[i])
                #add
                H[i, index:index] = [new_neuron]
            else:
                del H[i, index]
        return P
                

    @staticmethod
    def mutate_constants(population):
        p = Primitives.copy(population.select())
        if randint(0, 1):
            population.pressure = ( 
                    population.pressure + math.ceil(gauss(0, 3)))
            if population.pressure > 15:
                population.pressure = 15
            elif population.pressure < 4:
                population.pressure = 4
        p.speed = abs(p.speed + gauss(0, 0.2))
        return p

    @staticmethod
    def add_or_remove_layer(population):
        P = Primitives.copy(population.select())
        H = P.hidden_layers
        index = randint(0, len(H))
        if len(H) == 0 or randint(0, 1):
            #generate layer
            new = Primitives.initlayer(randrange(1, P.ninps), 
                    P.shape[index])
            #add:
            H.insert(index, new)
        else:
            del H[index]
        return P

