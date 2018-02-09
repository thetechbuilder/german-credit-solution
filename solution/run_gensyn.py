#! /usr/bin/env python3
import csv
from sys import path
from random import randrange, uniform, randint
#included modules:
path.append('nn')
path.append('ga')
from ga import GaOperators
from gensyn import Individual, RawOperators, Gensyn
from german_credit import tonargs
from gensyn import *
#initialize population
size = 50
generations = 100
data_set_file = 'data/GermanCredit.csv'
observations = csv.reader(open(data_set_file))
next(observations) #pass header
training_set = [tonargs(next(observations)) for x in range(600)]
validation_set = [tonargs(next(observations)) for x in range(400)]

input_count = len(training_set[0][0])
output_count = 1
P = []
#helpers:
def __appendlayer(l):
    sizes = l[-2], l[-1]
    l.insert(-1, 
            randint(abs(min(sizes) - randint(0, 2)) + 1, 
                (max(sizes) + randint(1, 2))))
print()
print("Initial populaiton:")
f = open("initial_pop.txt", 'w')
get_periods = lambda ll: 12000*(2**(len(ll)-2))
#initial population
for x in range(size):
    ll = [input_count, output_count]
    if randint(0, 4):
        __appendlayer(ll) #1
        if randint(0, 3):
            __appendlayer(ll) #2
            if not randint(0, 4):
                __appendlayer(ll) #3
                if not randint(0, 4):
                    __appendlayer(ll) #4
                    if not randint(0, 4):
                        __appendlayer(ll) #5
    print(ll)
    f.write(str(ll))
    P.append(Individual(ll, n = uniform(0.005, 1), 
        periods = get_periods(ll)))
f.close()
#initialize population:
P = Population(P, pressure = size//6)
#initialize operators:
O = GaOperators(
        [
            RawOperators.crossover_1, 
            RawOperators.reproduction, 
            RawOperators.add_and_remove_neuron,
            RawOperators.mutate_constants,
            RawOperators.add_or_remove_layer],
        [45, 15, 15, 10, 5])
#initialize neural network synthesis object:
G = Gensyn(P, O, training_set, validation_set)
#log writer
net_log = csv.writer(open('netprofit.csv', 'w'))
fp_log = csv.writer(open('fp.csv', 'w'))
fn_log = csv.writer(open('fn.csv', 'w'))
structure_log = csv.writer(open('sl.csv', 'w'))
speed_log = csv.writer(open('speed.csv', 'w'))
#insert headers
for x in [net_log, fp_log, fn_log, structure_log]:
    x.writerow(["GEN_NUMBER"] + list(range(size)))

input("HERE YOU GO")
for x in range(generations):
    print("\nGENERATION: %s"%x)
    G.evolve()
    for x in G.population:
        if randint(0, 3):
            x.periods = get_periods(x.shape)
    net_log.writerow([x] + [x.fitness[0] for x in G.population])
    fp_log.writerow([x] + [x.fitness[1] for x in G.population])
    fn_log.writerow([x] + [x.fitness[2] for x in G.population])
    structure_log.writerow(
            ([x] + [str(x.shape) for x in G.population]))
    speed_log.writerow(([x] + [x.speed for x in G.population]))

