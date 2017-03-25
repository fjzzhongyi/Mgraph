import os
import sys
import numpy as np
import copy
import time
from math import *
import math
import os
import multiprocessing
import time
import networkx as nx
from scipy.stats import norm

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                # print '%s: Exiting' % proc_name
                break
            # print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, graph, pw, nid):
        self.graph = graph
        self.pw = pw
        self.nid = nid
    def __call__(self):
        # this is the place to do your work
        route = dij(self.graph, self.pw, self.nid)
        return self.nid, route
    def __str__(self):
        return '%s' % (self.nid)


def calc_true_subgraph_score(att, true_subgraph):
    alphas = list(set([att[i][0] for i in true_subgraph]))
    score = None
    for alpha in alphas:
        s = f_score1(true_subgraph, att, alpha)
        if not score or s > score:
            score = s
    return s
     
 
def f_score1(subset, att, alpha = 0.05):
    n = len(subset) * 1.0
    nalpha = len([att[nid][0] for nid in subset if att[nid][0] <= alpha])
    return n * kl(nalpha / n, alpha)
     
     
def dij(G1, att, src):
    G = dict()
    for nid, neis in G1.items():
        if len(neis) > 0:
            G[nid] = neis
             
    if src not in G:
        raise TypeError('the root of the shortest path tree cannot be found in the G')
 
    previous = {}
    d = {}
    path = {}
 
    for v in G:
        d[v] = float('inf')
    d[src] = att[src]
    Q = {v:0 for v in G}
    while len(Q) > 0:
        temp = {v:d[v] for v in Q}
        u = min(temp, key=temp.get)
        # flag = False
        del Q[u]
        for v in G[u]:
            if v not in Q:
                continue
            new_distance = d[u] + att[v]
            if new_distance < d[v]:
                d[v] = new_distance
                previous[v] = u
    Q = [v for v in G if v != src]
    for v in Q:
        path[v] = [v]
        if previous.has_key(v):
            u = previous[v]
            while u != src:
                path[v].append(u)
                u = previous[u]
            path[v].append(u)
    path[src] = [src]
 
    return [d, path]
 
def printstat1(graph, att, src, route):
    print '--------------------------'
    print 'Graph :', graph
    print 'att   :', att
    print 'Source:', src
    distance = route[0]
    path = route[1]
    for i in graph:
        print '[', src, '->', i, '] distance:', distance[i]
        print '[', src, '->', i, '] path:', path[i]
        print
    print
 
def delta(pvalue, alpha):
    if pvalue <= alpha:
        return 1
    else:
        return -1
 
def calc_pathweight(graph, att): 
    N = len(att)
    pw = [0] * N
    for nid in range(N):
        if att[nid] > 0:
            pw[nid] = 0
        else:
            neis = graph[nid]
            posi_neis = [item for item in neis if att[item] > 0]
            if len(posi_neis) == 0:
                pw[nid] = -1 * att[nid]
            else:
                pw[nid] = -1 * min(0, -1 + sum([att[nid1] * 1.0 / len(graph[nid1]) for nid1 in posi_neis]))
    return pw
 
 
def topk_gain(shortpaths, topk, na, nb, npath):
    sdist = None
    snid1 = None
    snid2 = None
    spath = None
    for nid in topk:
        if nid not in [na, nb]:
            [nid1, dist] = min([[nid1, shortpaths[nid][0][nid1]] for nid1 in npath], key = lambda item: item[1])
            if not sdist or sdist > dist:
                sdist = dist
                snid1 = nid1
                snid2 = nid
    return snid1, snid2, sdist, shortpaths[snid2][1][snid1]
 
 
def gain_path(path, att):
    weights = [att[nid] for nid in path]
    return sum(weights) - max(weights)
 
 
def merge_set(graph, att, compdict, comp, glb_shortest_paths):
    neis = []
    n = len(graph)
    for i in comp:
        for j in graph[i]:
            if j not in comp:
                neis.append(j)
                graph[j] = [item for item in graph[j] if item not in comp]
                if n not in graph[j]:
                    graph[j].append(n)
        graph[i] = []
    graph[n] = list(set(neis))
    att.append(sum(att[i] for i in comp))
    compdict[n] = comp
 
def prec_recall(detect_subgraph, true_subgraph):
    n = 0.0
    for i in detect_subgraph:
        if i in true_subgraph:
            n += 1
    prec = n / len(detect_subgraph)
    recall = n / len(true_subgraph)
    fmeasure = 0.0
    if float(prec) < 1e-6 and float(recall) < 1e-6:
        fmeasure = 0.0
    else:
        fmeasure = 2*( prec*recall / (prec+recall) )
    return prec, recall, fmeasure
     
     
def s(graph, att, shortpaths, na, nb, nc):
#    print att
    N = len(att)
    mdist = None
    mnid = None
#    print 'N', N
    for nid in range(N):
        if nid not in [na, nb, nc] and graph[nid]:
            dist = sum([shortpaths[i][0][nid] for i in [na, nb, nc]])
            if not mdist or mdist > dist:
                mdist = dist
                mnid = nid
#    print 'nc, mnid', nc, mnid
    return mnid, mdist, list(set(shortpaths[na][1][mnid] + shortpaths[nb][1][mnid] + shortpaths[nc][1][mnid]))
 
 
def w(att, path):
    return sum([att[i] for i in path])







#--------------------------------------------------------------------------------------------------
#S contains the pvalue and id
def bjscore(subset, att,globalPValue, alpha_max = 0.15):
    S = []
    for nid in subset:
        S.append([nid, globalPValue[nid]])
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    nplus = 0.0
    nminus = 0.0
    for nid in subset:
        if att[nid] > 0:
            nminus += att[nid]
        else:
            nplus += 1
    return (nplus + nminus) * KL(nminus / (nplus + nminus), alpha)
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

def hcscore(subset, att,globalPValue, alpha_max = 0.15):
    S = []
    for nid in subset:
        S.append([nid, globalPValue[nid]])
    #alpha = max(item[1] for item in S)
    #if alpha > alpha_max:
    #    alpha = alpha_max
    alpha = alpha_max   
    nplus = 0.0
    nminus = 0.0
    for nid in subset:
        if att[nid] > 0:
            nminus += att[nid]
        else:
            nplus += 1
    N_alpha = nplus
    term = (N_alpha*1.0/len(subset)*1.0 - alpha)/ math.sqrt(alpha * (1 - alpha))
    return (math.sqrt(len(subset)))*term
  
def tippettScore(subset,att,globalPValue, alpha = 0.15):
    minPValue = 1.1 ;
    for nid in subset:
        if globalPValue[nid] < minPValue:
            minPValue = globalPValue[nid]
    if minPValue == 1.1:
        print 'pvalue error ...'
        sys.exit()
        return
    else:
        return -minPValue

def simesScore(subset, att,globalPValue, alpha = 0.15):
    subset = sorted(subset)
    S = []
    for nid in subset:
        S.append([nid, globalPValue[nid]])
    S = sorted(S, key=lambda xx:xx[1])
    minValue = 1.1
    for i in range(len(S)):
        AddI = i + 1
        term = S[i][1]/(AddI+0.0)
        if term < minValue:
            minValue = term
    if minValue == 1.1:
        print 'pvalue error ...'
        sys.exit()
        return
    else:
        return len(S)*minValue

def fisherScore(subset, att,globalPValue, alpha = 0.15):
    S = []
    for nid in subset:
        S.append([nid,globalPValue[nid]])
    sum = 0.0
    for item in S:
        sum = sum + np.log(item[1])
    return -(sum/len(S))

def stoufferScore(subset, att,globalPValue, alpha = 0.15):
    S = []
    for nid in subset:
        S.append([nid, globalPValue[nid]])
    sum = 0.0
    for item in S:
        sum = sum + norm.ppf(1-item[1])
    return -sum/np.sqrt(len(S)*1.0)

def pearsonScore(subset, att,globalPValue, alpha = 0.15):
    S = []
    for nid in subset:
        S.append([nid, globalPValue[nid]])
    maxValue = max(item[1] for item in S)
    if len(S) < 1:
        print 'the subset is null...'
        sys.exit()
    return - maxValue/(  math.pow( alpha, 1/(len(S)*1.0) )   )

def edgingtonScore(subset, att,globalPValue, alpha = 0.15):
    S =[]
    for nid in subset:
        S.append([nid,globalPValue[nid]])
    sum = 0.0
    for item in S:
        sum = sum + item[1]
    return -sum/(len(S)*1.0)

def npss_score(subset, att, npss,globalPValue, alpha = 0.15):
    if npss == 'BJ':
        return bjscore(subset, att,globalPValue, alpha)
    elif npss == 'HC':
        return hcscore(subset, att,globalPValue, alpha)
    elif npss == 'Edgington':
        return edgingtonScore(subset, att,globalPValue, alpha)
    elif npss == 'Pearson':
        return pearsonScore(subset,att,globalPValue, alpha)
    elif npss == 'Stouffer':
        return stoufferScore(subset,att,globalPValue, alpha)
    elif npss == 'Fisher':
        return fisherScore(subset, att,globalPValue, alpha)
    elif npss == 'Simes':
        return simesScore(subset, att,globalPValue, alpha)
    elif npss == 'Tippett':
        return tippettScore(subset, att,globalPValue, alpha)
    else:
        print 'npss score fails.'
        sys.exit()
        return None
#--------------------------------------------------------------------------------------------------











def refine_graph(graph1, att1, alpha):
    graph = copy.deepcopy(graph1)
    att = copy.deepcopy(att1)
    bigcomp = []
    comps = []
    for nid, neis in graph.items():
        if att[nid] > 0 and nid not in bigcomp:
            comp = [nid]
            queue = [nid]
            while queue:
                i = queue.pop()
                for j in graph[i]:
                    if att[j] > 0 and j not in comp:
                        queue.append(j)
                        comp.append(j)
                        bigcomp.append(j)
            comps.append(comp)
    comps = [comp for comp in comps if len(comp) > 1]
    n = len(graph)
    compdict = dict()
    for comp in comps:
        neis = []
        for i in comp:
            for j in graph[i]:
                if j not in comp:
                    neis.append(j)
                    graph[j] = [item for item in graph[j] if item not in comp]
                    if n not in graph[j]:
                        graph[j].append(n)
            graph[i] = []
        graph[n] = neis
        att.append(len(comp))
        compdict[n] = comp
        n += 1
    return graph, att, compdict


def additive_graphscan(graph, att, npss,globalPValue, ncores = 10, iterations_bound = 10, minutes = 30, pri = False):
    ori_graph = graph
    ori_att = att
    alpha_max = 0.15
    alphas = list(set([pvalue[0] for nid, pvalue in enumerate(att) if pvalue[0] <= alpha_max]))
    sstar = None
    sfscore = None
    salpha = None
    for alpha in alphas[:]:
        print alpha
        att1 = [delta(pvalue[0], alpha) for pvalue in ori_att]
        graph, att1, compdict = refine_graph(ori_graph, att1, alpha)
        print '# connected components: ', len(compdict)
        alpha_sstar, alpha_sfscore = additive_graphscan_proc(graph, att1,npss,globalPValue, compdict, alpha, iterations_bound, ncores, minutes, pri)
        if alpha_sfscore and (sstar == None or sfscore < alpha_sfscore):
            sfscore = alpha_sfscore
            sstar = alpha_sstar
            salpha = alpha
    return [sstar, sfscore, salpha]

     
def recover_subgraph(compdict, sstar):
    if sstar:
        oriset = []
        for i in sstar:
            if compdict.has_key(i):
                queue = [i]
                while queue:
                    nid = queue.pop()
                    for nid1 in compdict[nid]:
                        if compdict.has_key(nid1):
                            queue.append(nid1)
                        else:
                            oriset.append(nid1)
            else:
                oriset.append(i)
        sstar = list(sorted(oriset))
    return sstar


def additive_graphscan_proc(graph, att ,npss ,globalPValue, compdict, alpha, iterations_bound = 10, ncores = 7, minutes = 2, pri = False):
    flag = True
    sfset = None
    sfscore = None
    mset = []
    iteration = 0
    if len([nid for nid, neis in graph.items() if len(neis) > 0]) == 0:
        sstar = recover_subgraph(compdict, [len(graph)-1])
        sfscore = npss_score(sstar, att, npss,globalPValue, alpha)
        return sstar, sfscore
    [nid, score] = max(enumerate(att), key=lambda item: item[1])
    sfset = recover_subgraph(compdict, [nid])
    sfscore = npss_score(sfset, att, npss,globalPValue, alpha)
    glb_shortest_paths = {}
    start_time = time.time()
    while flag:
        if iteration % 2 == 0:
            print '****** iteration: ', iteration, ' ******'
            duration = (time.time() - start_time)
            if duration > minutes * 60:
                print 'time limit reached .....'
                break
        if iteration > iterations_bound:
            print 'iteration bound reached .....'
            break
        N = len([nid for nid, neis in graph.items() if len(neis) > 0])
        K = max(int(sqrt(N)), 10)
        if K > 10:
            K = 10
        nids = sorted([nid for nid in range(len(graph)) if len(graph[nid]) > 0 and att[nid] >= 1], key = lambda item: att[item] * -1)
        topk = list(nids[:K])
        if len(topk) == 0:
            break
        elif len(topk) == 1:
            sstar = recover_subgraph(compdict, topk)
            fs = npss_score(sstar, att, npss,globalPValue, alpha)
            if fs > sfscore:
                sfscore = fs
                mset = sstar
            break
        pw = calc_pathweight(graph, att)
        shortpaths = {}
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        # Start consumers
        num_consumers = ncores # We only use 5 cores.
        # print 'Creating %d consumers' % num_consumers
        consumers = [ Consumer(tasks, results) for i in xrange(num_consumers) ]
        for w in consumers:
            w.start()
        num_jobs = len(topk)
        # Enqueue jobs
        for nid in topk:
            tasks.put(Task(graph, pw, nid))
        # Add a poison pill for each consumer
        for i in xrange(num_consumers):
            tasks.put(None)
        fin = 0
        # Start printing results
        while num_jobs:
            nid, route = results.get()
            shortpaths[nid] = route
            num_jobs -= 1
        pstar = None
        pdist = None
        for nid1 in topk: 
            for nid2 in topk:
                if nid2 != nid1:
                    dist = shortpaths[nid1][0][nid2]
                    if not pdist or pdist > dist:
                        pstar = shortpaths[nid1][1][nid2]
                        pdist = dist
        if isinf(pdist):
            print '***** all infinite distances *******'
            break
        na = pstar[0]
        nb = pstar[-1]
        if len(topk) == 2:
            merge_set(graph, att, compdict, pstar, glb_shortest_paths)
            mset = recover_subgraph(compdict, pstar)
            flag = False
        else:
            [snid1, nc, sdist, spath] = topk_gain(shortpaths, topk, na, nb, pstar)
            g = gain_path(list(set(pstar + spath)), att) - max(att[nc], gain_path(pstar, att))
            if g <= 0:
                merge_set(graph, att, compdict, pstar, glb_shortest_paths)
                mset = recover_subgraph(compdict, pstar)
                if gain_path(pstar, att) <= 0:
                    flag = False
            else:
                pstarstar = list(set(pstar + spath))
                nid, ndist, npath = s(graph, att, shortpaths, na, nb, nc)
                w1 = sum([att[i] for i in npath])
                w2 = sum([att[i] for i in pstarstar])
                if w1 <= 0 and w2 <= 0:
                    flag = False
                else:
                    if w1 > w2:
                        merge_set(graph, att, compdict, npath, glb_shortest_paths)
                        mset = recover_subgraph(compdict, npath)
                    else:
                        merge_set(graph, att, compdict, pstarstar, glb_shortest_paths)
                        mset = recover_subgraph(compdict, pstarstar)
        if compdict or mset:
            if not sfset:
                sfscore = npss_score(mset, att, npss,globalPValue, alpha)
                sfset = mset
            else:
                score = npss_score(mset, att, npss,globalPValue, alpha)
                if sfscore < score:
                    sfset = mset
                    sfscore = score
        iteration += 1
    return sfset, sfscore

def printstat(graph, att, true_subgraph, subset_score, alpha = 0.01):
    if set(subset_score[0]) == set(true_subgraph):
        prec,recall,fmeasure = prec_recall(subset_score[0], true_subgraph)
    else:
        prec, recall,fmeasure = prec_recall(subset_score[0], true_subgraph)
    return prec,recall, fmeasure

def read_APDM_data(path):
    att = []
    pvalue = []
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.strip() == 'NodeID Weight':
            n = idx + 1
            break
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            att.append([float(items[1])])
            pvalue.append(float(items[1]))
    graph = {}
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            n1 = int(items[0])
            n2 = int(items[1])
            # if n1 == 38 or n2 == 38:
            #     print n1, n2
            if graph.has_key(n1):
                if n2 not in graph[n1]:
                    graph[n1].append(n2)
            else:
                graph[n1] = [n2]
            if graph.has_key(n2):
                if n1 not in graph[n2]:
                    graph[n2].append(n1)
            else:
                graph[n2] = [n1]
    true_subgraph = []
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            break
        else:
            items = line.split(' ')
            true_subgraph.append(int(items[0]))
            true_subgraph.append(int(items[1]))
    true_subgraph = sorted(list(set(true_subgraph)))
    return graph, att,pvalue, true_subgraph


def read_APDM_data_trans(path):
    att = []
    pvalue = []
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith('NodeID Weight'):
            n = idx + 1
            break
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            att.append([float(items[1])])
            pvalue.append(float(items[1]))
    graph = {}
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            n1 = int(items[0])
            n2 = int(items[1])
            # if n1 == 38 or n2 == 38:
            #     print n1, n2
            if graph.has_key(n1):
                if n2 not in graph[n1]:
                    graph[n1].append(n2)
            else:
                graph[n1] = [n2]
            if graph.has_key(n2):
                if n1 not in graph[n2]:
                    graph[n2].append(n1)
            else:
                graph[n2] = [n1]
    return graph, att,pvalue

def test_single_case(APDMFileName,npss,outPutFileName,noiseLevel='none',hour = 'none',alpha_max = 0.15,verbose_level = 0):
    start_time = time.time()
    if verbose_level <= 1:
        print '--------------------------------------------------------------------------------'
        print 'processing file : ', os.path.join(APDMFileName)
    if verbose_level == 1:
        print 'processing score : ', npss
    graph, att,globalPValue, true_subgraph = read_APDM_data(APDMFileName)
    subset_score = additive_graphscan(graph, att,npss,globalPValue)
    prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
    if verbose_level <= 1:
        print '--------------------------------------------------------------------------------'
    bjscore = subset_score[1]
    timeV = time.time() - start_time
    file = open("./additive_result_source_12500_data_min_30.txt",'a')
    file.write("{0:.6f}".format(prec)+"\t"+"{0:.6f}".format(recall)+"\t"+"{0:.6f}".format(fmeasure)+"\t"+
   "{0:.6f}".format(bjscore)+"\t"+"{0:.6f}".format(timeV)+"\t"+hour+"\t" +noiseLevel+"\t"+'source_12500'+"\n")
    file.close()
    
def test_water_pollution_data_noise_x(x,hours):
    folderName = '../../realDataSet/WaterData'
    subFolderNames = os.listdir(folderName)
    subFolderNames = [fileName for fileName in subFolderNames if os.path.isdir(folderName+'/'+fileName) ]
    for eachSubFolder in subFolderNames:
        print eachSubFolder
        if eachSubFolder == 'source_12500':
            for eachFileData in os.listdir(os.path.join(folderName,eachSubFolder)):
                noiseLevel = str(eachFileData.split("_")[5].split(".")[0])
                hour = str(eachFileData.split("_")[2])
                if noiseLevel == x and hour in hours:
                    print 'current processing file : ', os.path.join(folderName,eachSubFolder,eachFileData)
                    start_time = time.time()
                    data_source = str(eachSubFolder)
                    fileName = str(eachFileData)
                    npss = 'BJ'
                    graph, att,globalPValue, true_subgraph = read_APDM_data(os.path.join(folderName,eachSubFolder,eachFileData))
                    subset_score = additive_graphscan(graph, att,npss,globalPValue)
                    prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
                    bjscore = subset_score[1]
                    timeV = time.time() - start_time
                    hour = str(eachFileData.split("_")[2])
                    noiseLevel = str(eachFileData.split("_")[5].split(".")[0])
                    data_source = str(data_source)
                    file = open("./additive_result_min_30_12500.txt",'a')
                    file.write("{0:.6f}".format(prec)+"\t"+"{0:.6f}".format(recall)+"\t"+"{0:.6f}".format(fmeasure)+"\t"+
                   "{0:.6f}".format(bjscore)+"\t"+"{0:.6f}".format(timeV)+"\t"+hour+"\t"+ noiseLevel+"\t"+'source_12500'+"\n")
                    file.close()
   
def test_grid_data(folderName,data):
    for eachFileData in os.listdir(folderName):
        print 'processing file : ', os.path.join(folderName,eachFileData)
        alpha_max = 0.15
        npssArr = ['BJ','HC','Edgington','Pearson','Stouffer','Fisher','Simes','Tippett']
        for npss in npssArr:
                print 'processing score : ', npss
                start_time = time.time()
                graph, att,pvalue,true_subgraph = read_APDM_data(os.path.join(folderName,eachFileData))
                subset_score = additive_graphscan(graph, att, npss,pvalue)
                prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
                
                noiseLevel = str(eachFileData.split("_")[1].split(".")[0])
                timeV = str(time.time() - start_time)
                score = str(subset_score[1])
                file = open("./additive_result/additive_subgraph_result_grid_100_"+data+".txt",'a')
                file.write(str(prec)+" "+str(recall)+" "+str(fmeasure)+" "+score + " "+ timeV + " "+noiseLevel+" "+npss+"\n")
                file.close()
    print 'finish'
    return

def test_whole_grid_data():
    data = ['data1','data2','data3','data4',]
    data = ['data5','data6','data7','data8',]
    data = ['data9','data10','data11','data12',]
    data = ['data13','data14','data15','data16',]
    data = ['data17','data18','data19','data20']
    for dataset in data:
        test_grid_data('../simuDataSet/'+dataset,dataset)
    return

def test_whole_water():
    hours = ['03','04','05','06','07','08']
    test_water_pollution_data_noise_x('8',hours)
    return

def test_water_dataSet():
    rootFolderName = '../../realDataSet/WaterData/source_12500/'
    from multiprocessing import Pool
    pool = Pool(processes=10)
    for eachDataFile in os.listdir(rootFolderName):
        outPutFileName = './kddGreedy_result_source_12500.txt'
        noiseLevel = int(eachDataFile.split('_')[5].split('.')[0])
        hour = str(eachDataFile.split('_')[2])
        InputAPDM = os.path.join(rootFolderName,eachDataFile)
        print 'processing file : ',InputAPDM
        pool.apply_async(test_single_case_water, args=(InputAPDM, 'BJ', outPutFileName,noiseLevel,hour,) )
    pool.close()
    pool.join()

def test_single_case_water(APDMFileName,npss,outPutFileName,noiseLevel='none',hour = 'none',alpha_max = 0.15,verbose_level = 0):
    start_time = time.time()
    if verbose_level <= 1:
        print '--------------------------------------------------------------------------------'
        print 'processing file : ', os.path.join(APDMFileName)
    if verbose_level == 1:
        print 'processing score : ', npss
    graph, att,globalPValue, true_subgraph = read_APDM_data(APDMFileName)
    subset_score = additive_graphscan(graph, att,npss,globalPValue)
    prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
    if verbose_level <= 1:
        print '--------------------------------------------------------------------------------'
    bjscore = subset_score[1]
    timeV = time.time() - start_time
    file = open("./additive_result_source_12500_data_min_30.txt",'a')
    file.write("{0:.6f}".format(prec)+"\t"+"{0:.6f}".format(recall)+"\t"+"{0:.6f}".format(fmeasure)+"\t"+
   "{0:.6f}".format(bjscore)+"\t"+"{0:.6f}".format(timeV)+"\t"+hour+"\t" +noiseLevel+"\t"+'source_12500'+"\n")
    file.close()
    
def testing_single_case_trans(APDMFileName,npss,outPutFileName, alpha_max = 0.15,verbose_level = 0):
    start_time = time.time()
    if verbose_level == 1:
        print '--------------------------------------------------------------------------------'
        print 'processing file : ', os.path.join(APDMFileName)
    if verbose_level == 1:
        print 'processing score : ', npss
    #graph : adj list dict() ; att : list ; globalPValue : list
    graph, att,globalPValue = read_APDM_data_trans(APDMFileName)
    g = nx.Graph()
    g.add_nodes_from(range(1912))
    for item in graph:
        for nei in graph[item]:
            g.add_edge(item,nei)
    nodes = []
    for itemGraph in nx.connected_components(g):
        nodes = itemGraph
        break
    mapPValues = []
    for node in nodes:
        mapPValues.append(globalPValue[node])
    #update pvalue
    globalPValue = mapPValues
    new_graph = dict()
    for item in graph:
        if item in nodes:
            new_graph[nodes.index(item)] = []
            for nei in graph[item]:
                if nei in nodes:
                    new_graph[nodes.index(item)].append(nodes.index(nei))
    graph = new_graph #update graph
    new_att = [att[item] for item in nodes]
    att = new_att
    print 'current processing file : ', APDMFileName
    start_time = time.time()
    subset_score = additive_graphscan(graph, att,npss,globalPValue)
    bjscore = subset_score[1]
    resultSubSet = subset_score[0]
    timeV = time.time() - start_time
    file = open(outPutFileName,'a')
    id = APDMFileName.split('/')[5].split('.')[0]
    truenodes = [item for item in globalPValue if item <=0.15]
    file.write("{0:.6f}".format(bjscore)+"\t"+"{0:.6f}".format(timeV)+"\t"+str(len(resultSubSet))+
               "\t"+str(len(truenodes))+"\t"+id+"\n")
    file.close()
    
def test_trans():
    rootFolderName = '../../realDataSet/TransportationCandidate2'
    from multiprocessing import Pool
    pool = Pool(processes=1)
    for eachDataSet in os.listdir(rootFolderName):
        for eachDataFile in os.listdir(rootFolderName+'/'+eachDataSet):
            outPutFileName = '../../transportationDataResults/Additive_2013-07_result.txt'
            InputAPDM = rootFolderName+'/'+eachDataSet+'/'+eachDataFile
            print 'processing file: ',InputAPDM
            pool.apply_async(testing_single_case_trans, args=(InputAPDM,'BJ',outPutFileName))

def test_1():
    hours = ['01','02','03','04','05','06','07','08']
    test_water_pollution_data_noise_x('0',hours)
    test_water_pollution_data_noise_x('2',hours)
    test_water_pollution_data_noise_x('4',hours)
def test_2():
    hours = ['01','02','03','04','05','06','07','08']
    test_water_pollution_data_noise_x('6',hours)
    test_water_pollution_data_noise_x('8',hours)
    test_water_pollution_data_noise_x('10',hours)
def test_3():
    hours = ['01','02','03','04','05','06','07','08']
    test_water_pollution_data_noise_x('12',hours)
    test_water_pollution_data_noise_x('14',hours)
    test_water_pollution_data_noise_x('16',hours)
def test_4():
    hours = ['01','02','03','04','05','06','07','08']
    test_water_pollution_data_noise_x('18',hours)
    test_water_pollution_data_noise_x('20',hours)
    test_water_pollution_data_noise_x('22',hours)
def test_5():
    hours = ['01','02','03','04','05','06','07','08']
    test_water_pollution_data_noise_x('24',hours)
    test_water_pollution_data_noise_x('26',hours)
    test_water_pollution_data_noise_x('28',hours)
def test_6():
    hours = ['01','02','03','04','05','06','07','08']
    test_water_pollution_data_noise_x('30',hours)
def main():
    test_1()
    return
if __name__ == "__main__":
    main()