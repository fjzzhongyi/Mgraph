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
########################################
#  return [subgraph, F-socre]
#######################################

def find_connected_components(graph, subset, alpha = 0.15):
    #for item in graph:
    #    print item,graph[item]
    #time.sleep(1000)
    #print 'length of subset : ',len(subset),subset
    #time.sleep(1000)
    g = nx.Graph()
    #-----add nodes
    g.add_nodes_from(subset)
    for i_1 in subset:
        for i_2 in graph[i_1]:
            if i_2 in subset:
                g.add_edge(i_1,i_2)
    return [itemGraph for itemGraph in nx.connected_components(g)]
    
def npss_detection(PValue, E, alpha_max, npss,verbose_level = 0):#pvalue, network,alpha
    
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
    
    #K = []
    #itemsGraphNodes = find_connected_components(graph,candidate)
    #for itemGraph in itemsGraphNodes:
    #    if len(itemsGraphNodes) <= 5:
    #        K.append(itemGraph[random.randint(0,len(itemGraph)-1)])
    #    elif len(itemGraph) > 10:
    #        K.append(itemGraph[random.randint(0,len(itemGraph)-1)])
    #if len(K) < 5:
    #    K.append(itemsGraphNodes[0][0])
    #    K.append(itemsGraphNodes[0][1])
    #    K.append(itemsGraphNodes[0][2])
    #print 'size : ',len(candidate)
    #K = [candidate[k] for k in np.random.permutation(len(candidate))[0:int(0.3*len(candidate))] ]
    #print 'graphs size : size K : ',len(itemsGraphNodes),len(K)
    K = 100
    #for k in K:
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
            if len(maxS) == len(S):
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

def unit_test(verbose_level = 0):
    timeStart = time.time()
    testCase1 = '../../simuDataSet/data1/APDM-GridData-400-precen-0.1-noise_0.txt'
    graph,pvalue,true_subgraph = readAPDM.read_APDM_data(testCase1)
    resultNodes,score = npss_detection(pvalue, graph, 0.15, 'BJ')
    if verbose_level != 0:
        print 'result Nodes : ',[item[0] for item in resultNodes]
        print 'true_subgraph : ',true_subgraph
        print 'score : ',score
    resultSet = set([item[0] for item in resultNodes])
    trueSet = set([item for item in true_subgraph])
    if resultSet == trueSet:
        print 'test passed ...'
    else:
        print 'test failed ...'
    print 'running time : ',time.time() - timeStart
    
    
    timeStart = time.time()
    testCase2 = '../../simuDataSet/data2/APDM-GridData-400-precen-0.1-noise_0.txt'
    graph,pvalue,true_subgraph = readAPDM.read_APDM_data(testCase2)
    resultNodes,score = npss_detection(pvalue, graph, 0.15, 'BJ')
    if verbose_level != 0:
        print 'result Nodes : ',[item[0] for item in resultNodes]
        print 'true_subgraph : ',true_subgraph
        print 'score : ',score
    resultSet = set([item[0] for item in resultNodes])
    trueSet = set([item for item in true_subgraph])
    if resultSet == trueSet:
        print 'test passed ...'
    else:
        print 'test failed ...'
    print 'running time : ',time.time() - timeStart

def findAlphaSet(pvalue,alpha_max = 0.15):
    alphaSet = []
    from scipy.cluster.vq import kmeans,vq
    import numpy as np
    y = np.array([value for index,value in pvalue.items() if value < alpha_max])
    codebook,_=kmeans(y, 5)
    cluster_indices,_=vq(y, codebook)
    clusters = Set()
    for item in cluster_indices:
        clusters.add(item)
    minValues = dict()
    maxValues = dict()
    average = dict()
    for i in range(len(clusters)):
        minValues[i] = 1.1
        average[i] = []
        maxValues[i] = -1.1
    count = 0
    for item in y:
        if minValues[cluster_indices[count]] > item:
            minValues[cluster_indices[count]] = item
        if maxValues[cluster_indices[count]] < item:
            maxValues[cluster_indices[count]] = item
        average[cluster_indices[count]].append(item)
        count = count + 1
    aver = []
    for index,value in average.items():
        aver.append(np.mean(value))
    finalSet = [value for index,value in minValues.items()]
    finalSet.append(0.15)
    aa = []
    for item in finalSet:
        if item > 0.01:
            aa.append(item)
    return aa

def prec_recall(detect_subgraph, true_subgraph):
    n = 0.0
    for i in detect_subgraph:
        if i in true_subgraph:
            n += 1
    if detect_subgraph:
        prec = n / len(detect_subgraph)
    else:
        prec = 0
    recall = n / len(true_subgraph)
    fmeasure = 0.0
    if float(prec) < 1e-6 and float(recall) < 1e-6:
        fmeasure = 0.0
    else:
        fmeasure = 2*( prec*recall / (prec+recall) )
    return prec, recall, fmeasure

def test_grid_data(folderName,data,verbose_level = 0):
    for eachFileData in os.listdir(folderName):
        if verbose_level <= 1:
            print '--------------------------------------------------------------------------------'
            print 'processing file : ', os.path.join(folderName,eachFileData)
        
        npssArr = ['BJ','HC','Edgington','Pearson','Stouffer','Fisher','Simes','Tippett']
        npssArr = ['HC']
        npssArr = ['Tippett']
        npssArr = ['BJ','HC','KS','Smirnow']
        npssArr = ['CUSUM']
        npssArr = ['TailRun']
        npssArr = ['BJ','HC','KS','Smirnow','CUSUM','TailRun']
        alpha_max = 0.15
        
        for npss in npssArr:
            if verbose_level <= 1:
                print 'processing score : ', npss
            
            start_time = time.time()
            graph, pvalue,true_subgraph = readAPDM.read_APDM_data(os.path.join(folderName,eachFileData))
            if npss.startswith('HC') or npss.startswith('BJ') :#or npss.startswith('KS') or npss.startswith('Smirnow'):
                flag = True
                tmpBestResultNodes = []
                tmpBestScore = 0.0
                tmpBestAlpha = 0.0
                pvalueSet = findAlphaSet(pvalue)
                for alpha in pvalueSet:
                    resultNodes,score = npss_detection(pvalue, graph, alpha, npss)
                    if flag == True:
                        tmpBestResultNodes = resultNodes
                        tmpBestScore = score
                        tmpBestAlpha = alpha
                        flag = False
                        continue
                    if tmpBestScore < score:
                        tmpBestResultNodes = resultNodes
                        tmpBestScore = score
                        tmpBestAlpha = alpha
                resultNodes = tmpBestResultNodes
                score = tmpBestScore
                alpha = tmpBestAlpha
            else:
                resultNodes,score = npss_detection(pvalue, graph, alpha_max, npss)
            if verbose_level == 2:
                print 'best alpha : ',alpha
            if verbose_level <= 1:
                print '--------------------------------------------------------------------------------'
            timeV = time.time() - start_time
            prec, recall, fmeasure = prec_recall(resultNodes,true_subgraph)
            noiseLevel = str(eachFileData.split("_")[1].split(".")[0])
            dataSize = str(eachFileData.split("-")[2])
            file = open("./kddGreedy_resultV3/kdd_greedy_result_grid_"+dataSize+"_"+data+".txt",'a')
            #file = open("./kddgreedy_result/kdd_greedy_result_grid_100_HC.txt",'a')
            file.write("{0:.6f}".format(prec)+"\t"+"{0:.6f}".format(recall)+"\t"+"{0:.6f}".format(fmeasure)+"\t"+
                       "{0:.6f}".format(score)+"\t"+"{0:.6f}".format(timeV)+"\t"+ noiseLevel+"\t"+npss+"\n")
            file.close()
    print 'finish'
    return

def test_single_case(APDMfileName, npss, outPutFileName, noiseLevel = 'none',hour = 'none', alpha_max = 0.15,verbose_level = 0):
    
    start_time = time.time()
    if verbose_level == 1:
        print '--------------------------------------------------------------------------------'
        print 'processing file : ', os.path.join(APDMfileName)
    if verbose_level == 1:
        print 'processing score : ', npss
    graph, pvalue,true_subgraph = readAPDM.read_APDM_data(APDMfileName)

    if npss.startswith('HC') or npss.startswith('BJ') :#or npss.startswith('KS') or npss.startswith('Smirnow'):
        flag = True
        tmpBestResultNodes = []
        tmpBestScore = 0.0
        tmpBestAlpha = 0.0
        pvalueSet = findAlphaSet(pvalue)
        for alpha in pvalueSet:
            resultNodes,score = npss_detection(pvalue, graph, alpha, npss)
            if flag == True:
                tmpBestResultNodes = resultNodes
                tmpBestScore = score
                tmpBestAlpha = alpha
                flag = False
                continue
            if tmpBestScore < score:
                tmpBestResultNodes = resultNodes
                tmpBestScore = score
                tmpBestAlpha = alpha
        resultNodes = tmpBestResultNodes
        score = tmpBestScore
        alpha = tmpBestAlpha
    else:
        resultNodes,score = npss_detection(pvalue, graph, alpha_max, npss)
    if verbose_level == 2:
        print 'best alpha : ',alpha
    if verbose_level <= 1:
        print '--------------------------------------------------------------------------------'
    timeV = time.time() - start_time
    prec, recall, fmeasure = prec_recall(resultNodes,true_subgraph)
    print 'output file name : ',outPutFileName
    file = open(outPutFileName,'a')
    file.write("{0:.6f}".format(prec)   +"\t"+"{0:.6f}".format(recall)+"\t"+
               "{0:.6f}".format(fmeasure)+"\t"+"{0:.6f}".format(score)+"\t"+
               "{0:.6f}".format(timeV)+"\t"+str(hour) +'\t'+ str(noiseLevel) + "\t"+"source_12500"+"\n")
    file.close()

def test_grid_400_dataSet():
    rootFolderName = '../../simuDataSet/'
    data = ['data1','data2','data3','data4','data5','data6',]
    from multiprocessing import Pool
    pool = Pool(processes=1)
    for dataSet in data:
        EachDataSet = os.path.join(rootFolderName,dataSet)
        for eachDataCase in os.listdir(EachDataSet):
            print EachDataSet,eachDataCase
            npssArr = ['BJ','HC','KS','Smirnow','CUSUM','TailRun']
            for npss in npssArr:
                dataSize = str(eachDataCase.split("-")[2])
                outPutFileName = "./kddGreedy_resultV3/kdd_greedy_result_grid_"+dataSize+"_"+dataSet+".txt"
                noiseLevel = str(eachDataCase.split("_")[1].split(".")[0])
                APDMINPUT = os.path.join(rootFolderName,dataSet,eachDataCase)
                pool.apply(test_single_case, args=(APDMINPUT, npss, outPutFileName,noiseLevel,))
    pool.close()
    pool.join()
    print 'finish'

def test_water_dataSet():
    rootFolderName = '../../realDataSet/WaterData/source_12500/'
    from multiprocessing import Pool
    pool = Pool(processes=30)
    for eachDataFile in os.listdir(rootFolderName):
        npssArr =  ['BJ']#,'HC']#,'KS','Smirnow','CUSUM','TailRun']
        #npssArr = ['KS','Smirnow','CUSUM','TailRun']
        npssArr = ['BJ']
        for npss in npssArr:
            dataSet = 'source_12500'
            outPutFileName = './kddGreedy_result_'+dataSet+'.txt'
            noiseLevel = int(eachDataFile.split('_')[5].split('.')[0])
            hour = eachDataFile.split('_')[2]
            InputAPDM = os.path.join(rootFolderName,eachDataFile)
            print 'processing file : ',InputAPDM
            pool.apply_async(test_single_case, args=(InputAPDM, npss, outPutFileName,noiseLevel,hour,) )
    pool.close()
    pool.join()
    
    

def test_single_case_trans(APDMfileName, npss, outPutFileName, alpha_max = 0.15,verbose_level = 0):
    
    start_time = time.time()
    if verbose_level == 1:
        print '--------------------------------------------------------------------------------'
        print 'processing file : ', os.path.join(APDMfileName)
    if verbose_level == 1:
        print 'processing score : ', npss
    graph, pvalue = readAPDM_trans.read_APDM_data_transportation(APDMfileName)
    
    g = nx.Graph()
    #-----add nodes
    g.add_nodes_from(range(1912))
    for item in graph:
        items = item.split("_")
        g.add_edge(int(items[0]),int(items[1]))
    nodes = []
    for itemGraph in nx.connected_components(g):
        nodes = itemGraph
        break
    mapPValues = []
    for node in nodes:
        mapPValues.append(pvalue[node])
    #update pvalue
    pvalue = dict()
    count = 0
    for item in mapPValues:
        pvalue[count] = item
        count = count + 1
        
    new_graph = dict()
    for item in graph:
        items = item.split("_")
        if int(items[0]) not in nodes or int(items[1]) not in nodes:
            continue
        else:
            new_graph[item] = graph[item]
    new_graph2 = dict()
    for item in new_graph:
        items = item.split("_")
        x = nodes.index(int(items[0]))
        y = nodes.index(int(items[1]))
        new_graph2[str(x)+"_"+str(y)] = new_graph[item]
    graph = new_graph2  #update graph
    
    if npss.startswith('HC') or npss.startswith('BJ') :#or npss.startswith('KS') or npss.startswith('Smirnow'):
        flag = True
        tmpBestResultNodes = []
        tmpBestScore = 0.0
        tmpBestAlpha = 0.0
        pvalueSet = findAlphaSet(pvalue)
        for alpha in pvalueSet:
            resultNodes,score = npss_detection(pvalue, graph, alpha, npss)
            if flag == True:
                tmpBestResultNodes = resultNodes
                tmpBestScore = score
                tmpBestAlpha = alpha
                flag = False
                continue
            if tmpBestScore < score:
                tmpBestResultNodes = resultNodes
                tmpBestScore = score
                tmpBestAlpha = alpha
        resultNodes = tmpBestResultNodes
        score = tmpBestScore
        alpha = tmpBestAlpha
    else:
        resultNodes,score = npss_detection(pvalue, graph, alpha_max, npss)
        
    if verbose_level == 2:
        print 'best alpha : ',alpha
    if verbose_level <= 1:
        print '--------------------------------------------------------------------------------'
    timeV = time.time() - start_time
    print 'output file name : ',outPutFileName
    file = open(outPutFileName,'a')
    id = APDMfileName.split('/')[5].split('.')[0]
    truenodes = [item for item in pvalue if pvalue[item] <=0.15]
    file.write("{0:.6f}".format(score)+" "+"{0:.6f}".format(timeV)+ " "+str(len(resultNodes))
               +" "+str(len(truenodes))+" "+id+"\n")
    file.close()
    
def test_transportationData():
    rootFolderName = '../../realDataSet/TransportationCandidate2'
    from multiprocessing import Pool
    pool = Pool(processes=1)
    for eachDataSet in os.listdir(rootFolderName):
        for eachDataFile in os.listdir(rootFolderName+'/'+eachDataSet):
            outPutFileName = '../../transportationDataResults/kddGreedy_result_transportation.txt'
            InputAPDM = rootFolderName+'/'+eachDataSet+'/'+eachDataFile
            print 'processing file : ',InputAPDM
            pool.apply_async(test_single_case_trans, args=(InputAPDM, 'BJ', outPutFileName) )
    pool.close()
    pool.join()
    
def test_mode():
    testCase1 = '../../simuDataSet/data1/APDM-GridData-400-precen-0.1-noise_0.txt'
    npss = 'Smirnow'
    outPutFileName = '../../tmp/test.txt'
    test_single_case(testCase1, npss, outPutFileName)
if __name__ =='__main__':
    #test_grid_400_dataSet()
    #test_water_dataSet()
    #test_single_case_trans('../../realDataSet/TransportationCandidate2/2013-07-22/2013-07-22_33.txt','BJ','./kddGreedy_result_transportation.txt')
    test_transportationData()
    #test_mode()