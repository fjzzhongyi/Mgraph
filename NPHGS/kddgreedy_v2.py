import os
import sys
import random
import json
import math
from os.path import join
from math import log
import sys
import copy
sys.setrecursionlimit(10000)
epislon = 0.000001

########################################
#
#  return {date:[[subgraph], F-socre]}
#
#######################################

def deletePreviousOutput(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try: 
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

def bjscore(S, N, alpha_max):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    a = len([item for item in S if item[1] <= alpha]) / (len(S) * 1.0)
    b = alpha
    return N * (a * log(epislon + a/(b+epislon)) + (1-a)*log(epislon + (1-a)/(1-b+epislon)))

def hcscore(S, N, alpha_max):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    N_alpha = len([item for item in S if item[1] <= alpha])
    hc = (N_alpha - N * alpha) / math.sqrt(N * alpha * (1 - alpha))
    hc = max(0, hc)
    return hc
    
def npss_score(S, N, alpha_max, npss):
    if npss == 'BJ':
        return bjscore(S, N, alpha_max)
    elif npss == 'HC':
        return hcscore(S, N, alpha_max)
    else:
        return None
        
def npss_detection(place, E, alpha_max, npss):#pvalue, network,alpha
#    print place
    graph = {}
    for item in E:
        vertices = item.split('_')
        if vertices[0] not in graph:
            graph[vertices[0]] = {}
        graph[vertices[0]][vertices[1]] = E[item]
        if vertices[1] not in graph:
            graph[vertices[1]] = {}
        graph[vertices[1]][vertices[0]] = E[item]

    K = 5
    V = []
    S_STAR = []

   # print 'place '+str(place)
    for site, pv in place.items():
        V.append([site, pv])
    N = len(V)
    V = sorted(V, key=lambda xx:xx[1])#all node+pvalue base on p value decide priority node

    if len(V) < K:
        K = len(V)
    for k in range(K):#number of seed node
        S = {V[k][0]:V[k][1]}
        max_npss_score = npss_score([[item, S[item]] for item in S], N, alpha_max, npss)
        maxS = [[V[k][0], V[k][1]]]
        while(True):
            G = []
            for v1 in S:
                s_v1 = str(v1)
                if s_v1 not in graph:
                    continue
                for s_v2 in graph[s_v1]:
                    i_s_v2 = int(s_v2)
                    if i_s_v2 not in S:
                        G.append([i_s_v2, place[i_s_v2]]) 
            G = sorted(G, key=lambda xx:xx[1])
            #print S, 'G '+str(G)
            S1 = [[item, S[item]] for item in S]
            for item in G:
                S1.append(item)
                phi = npss_score(S1, N, alpha_max, npss)
                if phi >= max_npss_score:
                    maxS = copy.deepcopy(S1)
                    max_npss_score = phi

            if len(maxS) == len(S):
                break
            else:
                S = {item[0]:item[1] for item in maxS}
#        print 'maxS', maxS, 'maxbj', maxbj
        S_STAR.append([sorted(maxS, key = lambda item: item[0]), max_npss_score])# subgraphs and score

    S_STAR = sorted(S_STAR, key=lambda xx:xx[1])
    if len(S_STAR) > 0:
       # print 'star '+str(S_STAR)
        return S_STAR[len(S_STAR)-1]
    else:
        return None

def getinfnpss(pvalue, network, alpha_max, npss):
    subgraph = {}

    for eventdate, place in pvalue.items():
        S_STAR = npss_detection(place, network, alpha_max, npss)
        if S_STAR:
            subgraph[eventdate] = S_STAR
    
    return subgraph

def getinfnpss_f(alpha_max,graphfile, path, source, co1):
    pvaluefolder=path+'/output/'+source+'/'+co1+'/pvalue'
    subgraphfolder =  path+'/output/'+source+'/'+co1+'/subgraph'
    deletePreviousOutput(subgraphfolder)
    
#    files = [file for file in os.listdir(folder) if file.find(co1) >= 0]
    files = [file for file in os.listdir(pvaluefolder)]
    for file in files[:1]:
#        print file
        pvalue = {}
        for line in open(os.path.join(pvaluefolder, file)).readlines():
            item = line.split()
            pvalue[int(item[0])] = float(item[1])
        
        network = {}
        for line in open(graphfile).readlines():
            item = line.split()
            edge = '{0}_{1}'.format(item[0], item[1])
            network[edge] = float(item[2])
        
    
#        print 'Result subgraph: '+str(npss_detection(pvalue, network, alpha_max))
#        print '#########'
        '''
        for i in range(50):
            pvalue[i] = 0.01
    
        print 'Result 2: '
        
        print npss_detection(pvalue, network, alpha_max)
        '''
        print network
        print pvalue
        result=npss_detection(pvalue, network, alpha_max)
        print result 
        
        if not os.path.exists(subgraphfolder):
            os.makedirs(subgraphfolder)
        out = open(os.path.join(subgraphfolder, '{0}'.format(file)), 'w')
          
   
        for i in range(len(result[0])):            
            out.write('{0}'.format(result[0][i]) + ' ')
        out.write('{0}'.format(result[1]))
       # out.write('{0}'.format(result))
        out.close()

def unit_test():
    pvalue = {'2013-02-04':{0: 0.01, 1: 0.1, 2: 0.1, 3: 0.01, 4: 0.1},
              '2013-02-05':{0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.1},
              '2013-02-06':{0: 0.01, 1: 0.01, 2: 0.1, 3: 0.1, 4: 0.01}}
    network = {'0_1': 0.2, '0_2': 0.1, '0_3': 0.2, '0_4': 0.2}
    alpha_max = 1
    expected_results = {'2013-02-04': [[[0, 0.01], [3, 0.01]], 23.025351004943786],
                        '2013-02-05': [[[0, 0.01], [1, 0.01], [2, 0.01], [3, 0.01]], 23.025351004943786],
                        '2013-02-06': [[[0, 0.01], [1, 0.01], [4, 0.01]], 23.025351004943786]}
#    npss = 'BJ'
    npss = 'HC'
    results = getinfnpss(pvalue, network, alpha_max, npss)
    print results
    for dt, subgraph in results.items():
        expected_subgraph = expected_results[dt]
        if set([item[0] for item in subgraph[0]]) != set([item[0] for item in expected_subgraph[0]]):
            return False
    return True


if __name__ =='__main__':

    print unit_test()
