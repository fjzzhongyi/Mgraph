import os
import sys
import random
import json
import time
import math
from os.path import join
from math import log
from sets import Set
import sys
import numpy as np
from scipy.stats import norm

#S is a list each element has form [item[0] item[1]] : [nodeID pvalue]
def bjscore(S, alpha):
    abn = len([item for item in S if item[1] <= alpha])*1.0
    nn = (len(S) - abn)*1.0
    score = (nn + abn) * KL(abn/(nn + abn), alpha)
    return score

def KL(t, x):
    x = x * 1.0
    if 0 < x and x < t and t <= 1:
        if t >= 1:
            return t * np.log(t / x)
        else:
            return t * np.log(t / x) + (1 - t) * np.log((1 - t) / (1 - x))
    elif 0 <= t and t <= x and x <= 1:
        return 0
    else:
        print 'KL distance error, this must not occur.'
        sys.exit()
        return inf

#S is a list each element has form [item[0] item[1]] : [nodeID pvalue]
def hcscore(S, alpha):
    N_alpha = len([item for item in S if item[1] <= alpha])*1.0
    return math.sqrt( len(S)*1.0 ) * (N_alpha*1.0/len(S)*1.0 - alpha) / math.sqrt(alpha * (1 - alpha)) ;

#S is a list each element has form [item[0] item[1]] : [nodeID pvalue]
def tippettScore(S):
    return -min(item[1] for item in S)

#S is a list each element has form [item[0] item[1]] : [nodeID pvalue]
def simesScore(S):
    copyedS = copy.deepcopy(S)
    copyedS = sorted(copyedS,key=lambda xx:xx[1])#sort by pvalue
    minValue = 1.1
    for i in range(len(S)):
        AddI = (i + 1)*1.0
        term = copyedS[i][1] / AddI
        if term < minValue:
            minValue = term
    return len(S)*minValue

def fisherScore(S):
    sum = 0.0
    for item in S:
        if item[1] <= 0.0:
            item[1] = 1e-6
        sum = sum + np.log(item[1])
    return -(sum/len(S))

def stoufferScore(S,N,alpha_max):
    sum = 0.0
    for item in S:
        if item[1] >= 1.0:
            item[1] = 0.9999
        if item[1] <= 0.0:
            item[1] = 0.0001
        sum = sum + norm.ppf(1-item[1])
    return sum/np.sqrt(len(S)*1.0)

def pearsonScore(S,alpha):
    maxValue = max(item[1] for item in S)
    return - maxValue/(  math.pow( alpha, 1/(len(S)*1.0) )   )

def edgingtonScore(S):
    sum = 0.0
    for item in S:
        sum = sum + item[1]
    return -sum/(len(S)*1.0)

#K-S test
def kolmogorovSmirnow(S):
    #the inverse of cumulative distrition function value is [1 - p-value]
    numericalPValue = norm.ppf([1-item[1] for item in S])
    X = numericalPValue
    flag = True
    maxValue = 0.0
    for x in X:
        Fx = norm.cdf(x)
        Fnx = len([xx for xx in X if xx <= x])*1.0 / len(X)*1.0
        funcValue = Fx - Fnx
        if flag == True:
            flag = False
            maxValue = funcValue
        else:
            if funcValue > maxValue:
                maxValue = funcValue
    return maxValue

def smirnov(S):
    numericalPValue = norm.ppf([1-item[1] for item in S])
    X = numericalPValue
    flag = True
    maxValue = 0.0
    for x in X:
        if x > 0:
            Fnx = len([xx for xx in X if xx<= x])*1.0 / len(X)*1.0
            FnMinusx = len([xx for xx in X if xx<= -x])*1.0 / len(X)*1.0
            funcValue = 1 - Fnx - FnMinusx
            if flag == True:
                flag = False
                maxValue = funcValue
            else:
                if funcValue > maxValue:
                    maxValue = funcValue
    return maxValue

def cumulativeSum(S):
    numericalPValue = norm.ppf([1-item[1] for item in S])
    absValue = [abs(item) for item in numericalPValue]
    indices = sorted(range(len(absValue)),key=lambda k:absValue[k], reverse=True)
    newPValue = range(len(numericalPValue))
    count = 0
    for item in indices:
        newPValue[count] = numericalPValue[item]
        count = count + 1
    numericalPValue = newPValue
    Ks = np.sign(numericalPValue)
    X = numericalPValue
    flag = True
    maxValue = 0.0
    N=[i+1 for i in range(len(S))]
    for k in N:
        Sk = sum(Ks[0:k])
        term = Sk*1.0 / math.sqrt(k)
        if flag == True:
            maxValue = term
            flag = False
        else:
            if maxValue < term:
                maxValue = term
    if flag == True:
        print 'subset S is empty ...'
        sys.exit()
    return maxValue

def tailRun(S):
    numericalPValue = norm.ppf([1-item[1] for item in S])
    absValue = [abs(item) for item in numericalPValue]
    indices = sorted(range(len(absValue)),key=lambda k:absValue[k], reverse=True)
    newPValue = range(len(numericalPValue))
    count = 0
    for item in indices:
        newPValue[count] = numericalPValue[item]
        count = count + 1
    numericalPValue = newPValue
    Ks = np.sign(numericalPValue)
    flag = True
    maxValue = 0
    for l in range(len(Ks)):
        if flag == True:
            maxValue = l+1
            flag = False
        else:
            if Ks[l] > 0:
                maxValue = l+1
            else:
                break
    return maxValue

def npss_score(S, alpha, npss):
    if npss == 'BJ':
        return bjscore(S, alpha)
    elif npss == 'HC':
        return hcscore(S, alpha)
    elif npss == 'Edgington':
        return edgingtonScore(S, alpha)
    elif npss == 'Pearson':
        return pearsonScore(S, alpha)
    elif npss == 'Stouffer':
        return stoufferScore(S, alpha)
    elif npss == 'Fisher':
        return fisherScore(S, alpha)
    elif npss == 'Simes':
        return simesScore(S, alpha)
    elif npss == 'Tippett':
        return tippettScore(S, alpha)
    elif npss == 'KS':
        return kolmogorovSmirnow(S)
    elif npss == 'Smirnow':
        return smirnov(S)
    elif npss == 'CUSUM':
        return cumulativeSum(S)
    elif npss == 'TailRun':
        return tailRun(S)
    else:
        print 'npss score fails.'
        sys.exit()
        return None
if __name__ == "__main__":
    S = [[0, 0.457415],[1, 0.857898],[2, 0.141857],[3, 0.717196],[4, 0.848001],
         [5, 0.858223],[6, 0.048555],[7, 0.084376],[8, 0.271023],[9, 0.933652],
         [10, 0.641297],[11, 0.032846],[12, 0.026611],[13, 0.431413],
         [14, 0.392010],[15, 0.592460]]
    S1 = [[2, 0.141857],[6, 0.048555],[7, 0.084376],[8, 0.271023],[9, 0.933652],
         [10, 0.641297],[11, 0.032846],[12, 0.026611],[13, 0.431413],]
    S2 = [[2, 0.141857],[6, 0.048555],[7, 0.084376],[11, 0.032846],[12, 0.026611],[13, 0.431413],]
    alpha = 0.15
    #print npss_score(S, alpha, 'KS')
    print npss_score(S, alpha, 'Smirnow')
    #print npss_score(S1, alpha, 'KS')
    print npss_score(S1, alpha, 'Smirnow')
    #print npss_score(S2, alpha, 'KS')
    print npss_score(S2, alpha, 'Smirnow')
    #print npss_score(S, alpha, 'CUSUM')
    #print npss_score(S1, alpha, 'CUSUM')
    #print npss_score(S2, alpha, 'CUSUM')