import csv
from bisect import bisect

def tonargs(bank_args):
    B = list(map(int, bank_args))
    CHK_ACCT = [0]*4
    CHK_ACCT[B[1]] = 1
    
    HISTORY = [0]*5
    HISTORY[B[3]] = 1 

    AMOUNT = [0]*6
    AMOUNT[bisect([500, 1000, 2000, 5000, 10000], B[10])] = 1

    SAV_ACCT = [0]*5
    SAV_ACCT[B[11]] = 1

    EMPLOYMENT = [0]*5
    EMPLOYMENT[B[12]] = 1

    PRESENT_RESIDENT = [0]*5
    PRESENT_RESIDENT[B[19]] = 1

    JOB = [0]*4
    JOB[B[27]] = 1

    #52 inputs
    return (CHK_ACCT + [B[2]] + HISTORY + B[4:10] + AMOUNT + SAV_ACCT + 
            EMPLOYMENT + B[13:19] + PRESENT_RESIDENT + B[20:27] + 
            JOB + B[28:31]), B[31]

def net_profit(value, result):
    if round(value):
        if result:
            return 100
        return -500
    return 0
