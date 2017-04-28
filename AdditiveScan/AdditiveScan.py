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
sys.path.append("..")
from sparkcontext import *

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

# g(p) is the gain that would result from merging path p into a single node
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

# determines a fourth node, ns in the graph as a Steiner point for na nb
def s(graph, att, shortpaths, na, nb, nc):
    N = len(att)
    mdist = None
    mnid = None
    for nid in range(N):
        if nid not in [na, nb, nc] and graph[nid]:
            dist = sum([shortpaths[i][0][nid] for i in [na, nb, nc]])
            if not mdist or mdist > dist:
                mdist = dist
                mnid = nid
    return mnid, mdist, list(set(shortpaths[na][1][mnid] + shortpaths[nb][1][mnid] + shortpaths[nc][1][mnid]))

# w(n) is the real-valued weight of node n
def w(att, path):
    return sum([att[i] for i in path])

#--------------------------------------------------------------------------------------------------
#S contains the pvalue and id
def bjscore(subset, att,globalPValue, alpha_max = 0.15):
    S = []
    for nid in subset:
        S.append([nid, globalPValue[nid]])
    print S
    alpha = max(item[1] for item in S)
    print alpha
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
    print t,x    
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

def npss_score(subset, att, npss,globalPValue, alpha = 0.15):
    if npss == 'BJ':
        return bjscore(subset, att,globalPValue, alpha)
    else:
        print 'npss score fails.'
        sys.exit()
        return None
#--------------------------------------------------------------------------------------------------
#
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

def additive_graphscan(graph, att, npss='BJ', iterations_bound=10, ncores=8, minutes=30):
    globalPValue=att
    ori_graph = graph
    ori_att = att
    alpha_max = 0.15
    alphas = list(set([pvalue[0] for nid, pvalue in enumerate(att) if pvalue[0] <= alpha_max]))
    sstar = None
    sfscore = None
    salpha = None
    def spark_proc(alpha):    
        # INPUT: alpha, ori_att, ori_graph
        print 'processing alpha : ',alpha,' ; start to execute additive scan'
        att1 = [ 1 if pvalue[0] <= alpha else -1 for pvalue in ori_att]
        graph, att1, compdict = refine_graph(ori_graph, att1, alpha)#compdict component dictionary
        print '# connected components: ', len(compdict)
        print '# cores : ',ncores
        alpha_sstar, alpha_sfscore =  \
        additive_graphscan_proc(graph, att1,npss,globalPValue, compdict, alpha, iterations_bound, ncores, minutes)
        return (alpha,(alpha_sstar,alpha_sfscore))
    global sc
    sc.broadcast(ori_att)
    sc.broadcast(ori_graph)
    result=sc.parallelize(alphas)\
            .map(spark_proc)\
            .reduce(lambda a,b: a if a[1][1]>b[1][1] else b)
    salpha=result[0]
    sstar=result[1][0]
    sfscore=result[1][1]
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


def additive_graphscan_proc(graph, att ,npss ,globalPValue, compdict, alpha=0.15, iterations_bound=10, ncores=8, minutes=30):
    
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
    while flag: #while positive gain path merges exist 
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
        K = max(int(sqrt(N)), 10) #identify top-k positive nodes where k = sqrt(N)
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
        #compute path weights pw(n) for all nodes and create single-source shortest paths from each top-k node
        pw = calc_pathweight(graph, att) 
        shortpaths = {}
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        num_consumers = ncores # We only use 5 cores.
        print num_consumers, 'threads are running to compute shortest path'
        consumers = [ Consumer(tasks, results) for i in xrange(num_consumers) ]
        for w in consumers: w.start()# Start consumers
        num_jobs = len(topk)
        for nid in topk: tasks.put(Task(graph, pw, nid))    # Enqueue jobs
        for i in xrange(num_consumers): tasks.put(None)  # Add a poison pill for each consumer
        fin = 0
        
        print 'starting to calculate shortest path ....'
        while num_jobs:# Start printing results
            nid, route = results.get()
            shortpaths[nid] = route
            num_jobs -= 1
        print 'finishing to calculate shortest path ....'
        #compute highest gain path p^* and record end points as na and nb
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
        print 'nodes na , nb are : ',na,nb
        #compute g(ni,p*) for each remaining top-k node, ni determine highest gain node for p* and record as nc
        #if no positive gain exists between p* and any ni, then merge p* and restart
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
    print 'finishing ...'
    return sfset, sfscore


def prec_recall_fmeasure(detect_subgraph, true_subgraph):
    n = 0.0
#   print detect_subgraph
    for i in detect_subgraph:
        if i in true_subgraph:
            n += 1
    if detect_subgraph:
        prec = n / len(detect_subgraph)
    else:
        prec = 0.0
    recall = n / len(true_subgraph)
    fmeasure = 0.0
    if float(prec) < 1e-6 and float(recall) < 1e-6:
        fmeasure = 0.0
    else:
        fmeasure = 2.0*( prec*recall / (prec+recall) )
    return prec, recall, fmeasure

def read_APDM_data(path):
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
            #    print n1, n2
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
            if items[0].startswith('null'):
                break
            true_subgraph.append(int(items[0]))
            true_subgraph.append(int(items[1]))
    true_subgraph = sorted(list(set(true_subgraph)))
    return graph, att,pvalue, true_subgraph

name_node={}
num_name={}
name_num={}
    
def genG(froot):
    print 'genG...'
    graph={}
    # f1 one line: 1000432103
    f1=open(os.path.join(froot,'nodes.dat'),'r')
    # f2 a line:  2803301701 3022787727
    f2=open(os.path.join(froot,'edges.dat'),'r')
    
    line=0
    s=f1.readline()
    while len(s)>0:
        s=s.strip()
        name_num[s]=line
        line+=1
        s=f1.readline()
        
    s=f2.readline()
    while len(s)>0:
        s=s.strip().split(' ')
        n1=int(name_num[s[0]])
        n2=int(name_num[s[1]])
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
        s=f2.readline()
    f1.close()
    f2.close()
    return graph

def addPvalue(froot,slice):
    print 'add_pvalue...'
    att=[]
    Pvalue=[]
    f=open(os.path.join(froot,'pvalues.dat'),'r')
    s=f.readline()
    while len(s)>0:
        s=s.strip().split(' ')
        att.append([float(s[slice])])
        Pvalue.append(float(s[slice]))
        s=f.readline()
    #print Pvalue
    return att,Pvalue
'''
def addPvalue(froot,slice):
    print 'add_pvalue...'
    Pvalue={}
    f=open(os.path.join(froot,'pvalues.dat'),'r')
    line=0
    s=f.readline()
    while len(s)>0:
        s=s.strip().split(' ')
        Pvalue[str(line)]=float(s[slice])
        line+=1
        s=f.readline()
    #print Pvalue
    return Pvalue
''' 
def getSlices(froot):
    print 'getSlices'
    f=open(os.path.join(froot,'pvalues.dat'),'r')
    s=f.readline().strip().split(' ')
    print 'slices= '+str(len(s)-1)
    return len(s)

if __name__ == "__main__":
    froot = os.path.join(sys.argv[1],'input')
    outroot= os.path.join(sys.argv[1],'output')
    iterations_bound = 10
    ncores = 10
    minutes = 30
    npss='BJ'
    G=genG(froot)
    slices=getSlices(froot)
    result=[]
    
    startTime = time.time()
    print 'start processing : '
    for slice in range(1,slices):
        print 'slice=' +str(slice)+'...'
        att,Pvalue=addPvalue(froot,slice)
        subset_score= additive_graphscan(G, att,'BJ',Pvalue,iterations_bound,ncores,minutes)
        resultNodes=subset_score[0]
        score=subset_score[1]
        result.append((resultNodes,score))
    runningTime = time.time() - startTime
    print 'finishing ,duration: '+ str(runningTime)
    
    fw=open('AddictiveScan_'+datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')+'.txt','w+')
    for each in result:
        resultNodes=each[0]
        score=each[1]
        fw.write(str([str(each) for each in resultNodes]))
        fw.write('\n')
        fw.flush()
    fw.close()
