import os
import sys
import random
import json
import math
from os.path import join
from sets import Set
import sys
import numpy as np
import copy
import time
import npssScore as npssS
import read_APDM_data as readAPDM
import read_APDM_data_transportation as readAPDM_trans
import networkx as nx


def npss_detection(PValue, E, alpha_max, npss, verbose_level = 0):#pvalue, network,alpha

    if verbose_level == 2:
        print 'PValue : ',PValue # is a dictionary, i:p-valuei
        print 'E : ',E # is a dictionary, 'i_j':1.0
        print 'alpha_max : ',alpha_max
        print 'npss : ',npss

    graph = {}
    for item in E:
        vertices = item.split('_')
        i = int(vertices[0])
        j = int(vertices[1])
        if i not in graph:
            graph[i] = {}
        graph[i][j] = E[item]
        if j not in graph:
            graph[j] = {}
        graph[j][i] = E[item]
    
    V = []
    S_STAR = []
    for nid, pv in PValue.items(): V.append([nid, pv])
    V = sorted(V, key=lambda xx:xx[1])#all node+pvalue base on p value decide priority node
    '''
    graph is a adjacency list id: dictionary
    '''
    if verbose_level == 2:
        for index,value in graph.items():
            print 'index : ',index , 'adj : ',value
        for item in V:
            print item 
    candidate = [k[0] for k in V if k[1] <= alpha_max]
    
    count = 1
    for item in V:
        if item[1] <= 0.15: count = count + 1
        else : break
    K = count
    print count
    for k in range(K):
        initNodeID = V[k][0]
        initNodePvalue = V[k][1] 
        S = {initNodeID:initNodePvalue} #{nid:pvalue}
        max_npss_score = npssS.npss_score([[item, S[item]] for item in S], alpha_max, npss)
        maxS = [[initNodeID, initNodePvalue]]
        while(True):
            G = []
            for v1 in S:
                if v1 not in graph:
                    print 'could not happen ...'
                    sys.exit()
                    continue
                for v2 in graph[v1]:
                    if v2 not in S and v2 not in [item[0] for item in G]:
                        G.append( [  v2, PValue[v2]  ] )
            G = sorted(G, key=lambda xx:xx[1])
            S1 = [[item, S[item]] for item in S]
            for item in G:
                S1.append(item)
                phi = npssS.npss_score(S1, alpha_max, npss)
                if phi > max_npss_score:
                    maxS = copy.deepcopy(S1)
                    max_npss_score = phi
            if len(maxS) == len(S) :
                break
            else:
                S = {item[0]:item[1] for item in maxS}
            if verbose_level != 0:
                print 'len(S) : ',len(S), ' len(maxS) : ',len(maxS)
                print 'S : ',[item for item in S]
                print 'maxS : ',[item[0] for item in maxS]
        S_STAR.append([sorted(maxS, key = lambda item: item[0]), max_npss_score])# subgraphs and score
    S_STAR = sorted(S_STAR, key=lambda xx:xx[1])
    if verbose_level != 0:
        for item in S_STAR:
            print item
    '''return the maximum npss value '''
    if len(S_STAR) > 0:
        subGraph  = S_STAR[len(S_STAR)-1]
        return [item[0] for item in subGraph[0]] , subGraph[1]
    else:
        print 'kddgreedy fails, check the error'
        sys.exit()
        return None
    
def finalTest_NPHGS_waterNetwork(resultFileName,rootFolder,controlListFile):
    resultGraphFolder = '../../ComparePreRecROC/WaterNetwork_12500/NPHGS/graphResults/'
    folders = []
    files = []
    with open(controlListFile) as f:
        for eachLine in f.readlines():
            if eachLine.rstrip().endswith('.txt'):files.append(eachLine.rstrip())
            else: folders.append(eachLine.rstrip())
    files = ['APDM-Water-source-12500_time_08_hour_noise_0.txt',
             'APDM-Water-source-12500_time_08_hour_noise_2.txt', 'APDM-Water-source-12500_time_08_hour_noise_4.txt',
             'APDM-Water-source-12500_time_08_hour_noise_6.txt', 'APDM-Water-source-12500_time_08_hour_noise_8.txt',]
    
    for fileName in files:
        graph, pvalue,trueSubGraph = readAPDM.read_APDM_data_waterNetwork(os.path.join(rootFolder,fileName))
        alpha_max = 0.15
        npss = 'BJ'
        startTime = time.time()
        print 'start processing : ',fileName
        resultNodes,score = npss_detection(pvalue, graph, alpha_max, npss)
        print 'finishing ',fileName
        runningTime = time.time() - startTime
        resultSet = set(resultNodes)
        trueSubGraph = set(trueSubGraph)
        intersect = resultSet.intersection(trueSubGraph)
        pre = (len(intersect)*1.0) / (len(resultNodes)*1.0)
        rec = (len(intersect)*1.0) / (len(trueSubGraph)*1.0)
        fmeasure = 2.0*(pre*rec) / (pre+rec)
        
        resultNodes = set(resultNodes)
        trueSubGraph = set(trueSubGraph)
        intersect = resultNodes.intersection(trueSubGraph)
        truePositive = len(intersect)*1.0 ;
        falseNegative = len(trueSubGraph)*1.0 - truePositive ;
        falsePositive = len(resultNodes)*1.0 - len(intersect)*1.0 ;
        trueNegative = len(pvalue) - len(trueSubGraph) - falsePositive ;
        tpr = truePositive / (truePositive+falseNegative) ;
        fpr = trueNegative / (falsePositive + trueNegative) ;    
        
        file = open(resultFileName,'a')
        file.write("{0:.6f}".format(0.0)+"\t"+
                   "{0:.6f}".format(score)+"\t"+
                   "{0:.6f}".format(runningTime)+"\t"+
                   "{0:06d}".format(1000)+"\t"+
                   "{0:06d}".format(len(resultNodes))+"\t"+
                   fileName.split('.txt')[0]+"\t"+
                   "{0:.6f}".format(0.0)+"\t"+
                   "{0:.6f}".format(pre)+"\t"+
                   "{0:.6f}".format(rec)+"\t"+
                   "{0:.6f}".format(fmeasure)+"\t"
                   "{0:.6f}".format(tpr)+"\t"+
                   "{0:.6f}".format(fpr)+ "\n")
        file.close()
        
        try: os.stat(resultGraphFolder)
        except: os.mkdir(resultGraphFolder)
        f = open(os.path.join(resultGraphFolder,fileName),'w')
        if len(resultNodes) == 0:
            f.write('null')
            f.close()
        else:
            for node in resultNodes:
                id = node
                f.write(str(id)+'\n')
            f.close()
if __name__ =='__main__':
    resultFileName = '../../ComparePreRecROC/WaterNetwork_12500/NPHGS/waterNetworkResult_NPHGS.txt'
    rootFolder = '../../realDataSet/WaterData/source_12500'
    controlListFile = '../../ComparePreRecROC/WaterNetwork_12500/controlFileList.txt'
    finalTest_NPHGS_waterNetwork(resultFileName,rootFolder,controlListFile)