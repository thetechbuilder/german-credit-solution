from sys import path
path.append('nn')
import csv
from random import choice
from unittest import TestCase
#included modules:
from german_credit import *
from neuro import Network

file_name = "data/GermanCredit.csv"

def update_progress(progress):
    print('\r [{0}] {1}%'.format('#'*int((progress/10)), 
        progress), end='')

class TestGermanCredit(TestCase):
    def test_simple_learning(self):
        print()
        S = [
                ([1,1,0,0,0,1,1],1),
                ([1,1,1,0,0,1,1],1),
                ([1,1,0,0,0,1,1],1),
                ([0,0,1,1,1,1,0],0),
                ([0,0,1,1,1,0,0],0),
                ([0,0,0,1,1,0,0],0),]
        T = [1,1,1,0,0,0]
        periods = 10000

        N = Network((7, 10, 1), n = 10)
        E = 0
        prog = 0
        while prog < periods:
            prog+=1
            s = choice(S)
            E += N.learn(s[0], (s[1],))
            update_progress(round(prog/periods*100))
        update_progress(100)
        print()
        
        for i, s in enumerate(S):
            print("{}) {:.8f} -- {}".format(i, 
                next(N.evaluate(s[0])), s[1]))
            
    def ftest_german_credit_learning_1(self):
        count = 20
        observations = csv.reader(open(file_name))
        next(observations)
        selection = [tonargs(next(observations)) for x in range(count)]
        N = Network((51, 21, 1), n = 0.1)
        TestGermanCredit.__learnGC(N, selection, periods = 10000)

    def ftest_german_credit_learning_2(self):
        count = 60
        observations = csv.reader(open(file_name))
        next(observations)
        selection = [tonargs(next(observations)) for x in range(count)]
        
        N = Network((51, 26, 1), n = 1)
        TestGermanCredit.__learnGC(N, selection, 100000)

    @staticmethod
    def __learnGC(N, selection, periods):
        prog = 0
        E = 0
        print()
        while prog < periods:
            prog += 1
            s = choice(selection)
            E = N.learn(s[0], (s[1],))
            update_progress(int(round(prog/periods*100)))
        update_progress(100)
        print()
        P = 0

        #bias learning:
        #--------------
        #observations = csv.reader(open(file_name))
        #next(observations)
        #for x in range(60):
        #    next(observations)
        #selection = [tonargs(next(observations)) for x in range(25)]
        
        for i, s in enumerate(selection):
            t = s[1]
            val = next(N.evaluate(s[0]))
            p = net_profit(val, t)
            P+=p
            print("{}) {:.8f} -- {} net profit: {}".format(
                i + 1, val, t, p))
        print()
        print("TOTAL PROFIT:{}".format(P))
        f = open('result.txt','w')
        f.write(str(N[:]))
        f.close()

