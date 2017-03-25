
__author__ = 'fengchen'

import os
import sys
import time
import copy
import numpy as np

from math import *
from time import sleep

import thread
import threading
import multiprocessing
from multiprocessing import Pool

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
    alphas = list(set([att[i][0] for i in true_subgraph if att[i][0] < 0.15]))
    score = None
    for alpha in alphas:
        s = f_score1(true_subgraph, att, alpha)
#        print true_subgraph, att, alpha
        if score == None or s > score:
            score = s
    return s
     
 
def f_score1(subset, att, alpha = 0.05):
    n = len(subset) * 1.0
    nalpha = len([att[nid][0] for nid in subset if att[nid][0] <= alpha])
    return n * KL(nalpha / n, alpha)
     
     
def dij(G1, att, src):
    print 'begin shortest path, waiting for finishing'
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
            # print new_distance
            # if new_distance > 100:
            #     flag = True
            #     break
        # if flag:
        #     break
 
    Q = [v for v in G if v != src]
    for v in Q:
        path[v] = [v]
#        print 'current vertex:', v
        if previous.has_key(v):
            u = previous[v]
            while u != src:
                path[v].append(u)
                u = previous[u]
            path[v].append(u)
    path[src] = [src]
    print 'finish shortest path ...'
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
#    print 'calc_pathweight'
#    print 'graph', graph
#    print 'att', att
    N = len(att)
    pw = [0] * N
    for nid in range(N):
        if att[nid] > 0:
            pw[nid] = 0
        else:
            neis = graph[nid]
            posi_neis = [item for item in neis if att[item] > 0]
#            print 'nid, neis, posi_neis', nid, neis, posi_neis
             
            if len(posi_neis) == 0:
                pw[nid] = -1 * att[nid]
            else:
                pw[nid] = -1 * min(0, -1 + sum([att[nid1] * 1.0 / len(graph[nid1]) for nid1 in posi_neis]))
#    print 'pw', pw
    return pw

"""
INPUT
topk: top k seed (positive) nodes
na and nb: two nodes in topk (check paper about the selection of na and nb)
npath: The shortest path that connects na and nb  
shortpaths: shortest paths between nodes in the graph

OUTPUT: 
snid2: the node in topk - {na, nb} that is closest to npath
snid1: the node in npath that is closest to snid2
sdist: distance between snid2 and snid1
"""
def topk_gain(shortpaths, topk, na, nb, npath):
    sdist = None
    snid1 = None
    snid2 = None
    spath = None
#    print 'topk_gain', npath
#    print 'shortpaths', shortpaths
#    print 'shortpaths[2]', shortpaths[2]
    for nid in topk:
        if nid not in [na, nb]:
            [nid1, dist] = min([[nid1, shortpaths[nid][0][nid1]] for nid1 in npath], key = lambda item: item[1])
#            print 'min', nid, nid1, dist
            if sdist == None or sdist > dist:
                sdist = dist
                snid1 = nid1
                snid2 = nid
#                print 'sdist, snid1, snid2', sdist, snid1, snid2
#    print 'na, nb, npath', na, nb, npath
#    print 'shortpaths[nid][0][nid1]', shortpaths[2][0][0]
#    print 'shortpaths[nid][0][nid1]', shortpaths[8][0][9]
#    print snid1, snid2, sdist, shortpaths[snid2][1][snid1]
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
    # flag = True
    # for nid in comp:
    #     if glb_shortest_paths.has_key(nid):
    #         glb_shortest_paths[n] = glb_shortest_paths[nid]
    #         for nid1 in glb_shortest_paths[n]:
    #             if nid1 in comp:
    #                 glb_shortest_paths[n] = None
    #             else:
    #                 glb_shortest_paths[n][nid1] = [n] + [item for item in glb_shortest_paths[n][nid1] if item not in comp]
    #         flag = False
    # if flag:
    #     for
#    if len([for nid, neis in graph.items()])
 
 
def prec_recall(detect_subgraph, true_subgraph):
    n = 0.0
#    print detect_subgraph
    for i in detect_subgraph:
        if i in true_subgraph:
            n += 1
    prec = n / len(detect_subgraph)
    recall = n / len(true_subgraph)
    return prec, recall
     
     
def s(graph, att, shortpaths, na, nb, nc):
#    print att
    N = len(att)
    mdist = None
    mnid = None
#    print 'N', N
    for nid in range(N):
        if nid not in [na, nb, nc] and graph[nid]:
            dist = sum([shortpaths[i][0][nid] for i in [na, nb, nc]])
            if mdist == None or mdist > dist:
                mdist = dist
                mnid = nid
#    print 'nc, mnid', nc, mnid
    return mnid, mdist, list(set(shortpaths[na][1][mnid] + shortpaths[nb][1][mnid] + shortpaths[nc][1][mnid]))
 
 
def w(att, path):
    return sum([att[i] for i in path])
 
             
def kl(ratio, alpha):
    if ratio == 0:
        ratio += 0.0000000001
    elif ratio == 1:
        ratio -= 0.0000000001
    alpha += 0.0000000001
    return ratio * np.log(ratio / alpha) + (1 - ratio) * np.log((1 - ratio) / (1 - alpha))

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

 
# Each node has a single attribute p-value
def f_score(subset, att, alpha = 0.05):
    nplus = 0.0
    nminus = 0.0
    for nid in subset:
        # if att[nid] > 1:
        #     print '************************************************', subset, att[nid]
        if att[nid] > 0:
            nminus += att[nid]
        else:
            nplus += 1
    return (nplus + nminus) * KL(nminus / (nplus + nminus), alpha)


def refine_graph(graph1, att1, alpha):
    # return graph1, att1, {}
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
            # att[i][0] = 1
        graph[n] = neis
        att.append(len(comp))
        compdict[n] = comp
        n += 1
    # print 'compdict', compdict
    return graph, att, compdict


def additive_graphscan(graph, att, ncores = 10, iterations_bound = 10, minutes = 30, pri = False):
    ori_graph = graph
    ori_att = att
    alpha_max = 0.15
    alphas = list(set([pvalue[0] for nid, pvalue in enumerate(att) if pvalue[0] <= alpha_max]))
    sstar = None
    sfscore = None
    salpha = None
    print len(alphas),alphas
    alphaMax = [0.15]
    for alpha in alphaMax:
    #for alpha in alphas[:]:
#        print alpha
        att1 = [delta(pvalue[0], alpha) for pvalue in ori_att]
#        print att1
        graph, att1, compdict = refine_graph(ori_graph, att1, alpha)
#        print att1
#        print '# connected components: ', len(compdict)
        alpha_sstar, alpha_sfscore = additive_graphscan_proc(graph, att1, compdict, alpha, iterations_bound, ncores, minutes, pri)
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


def additive_graphscan_proc(graph, att, compdict, alpha, iterations_bound = 10, ncores = 10, minutes = 30, pri = False):
    flag = True
    sfset = None
    sfscore = None
    mset = []
    iteration = 0
    if len([nid for nid, neis in graph.items() if len(neis) > 0]) == 0:
        sstar = recover_subgraph(compdict, [len(graph)-1])
        sfscore = f_score(sstar, att, alpha)
        return sstar, sfscore
    [nid, score] = max(enumerate(att), key=lambda item: item[1])
    sfset = recover_subgraph(compdict, [nid])
    sfscore = f_score(sfset, att, alpha)
    glb_shortest_paths = {}
    start_time = time.time()
    while flag:
        if iteration % 2 == 0:
            print '*************** iteration: ', iteration
            duration = (time.time() - start_time)
            if duration > minutes * 60:
                print 'time limit reached.....'
                break
        if iteration > iterations_bound:
            print 'iteration bound reached.....'
            break
        N = len([nid for nid, neis in graph.items() if len(neis) > 0])
        K = max(int(sqrt(N)), 10)
        if K > 10:
            K = 10
#        print att
#        print 'k', K, N
        nids = sorted([nid for nid in range(len(graph)) if len(graph[nid]) > 0 and att[nid] >= 1], key = lambda item: att[item] * -1)
        topk = list(nids[:K])
#        print 'top', nids
        if len(topk) == 0:
            break
        elif len(topk) == 1:
            sstar = recover_subgraph(compdict, topk)
            fs = f_score(sstar, att, alpha)
            if fs > sfscore:
                sfscore = fs
                mset = sstar
            break
        pw = calc_pathweight(graph, att) # the path weight of each node is defined in the paper
#        print 'pw', pw
        shortpaths = {}

        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()

        # Start consumers
        num_consumers = ncores # We only use 5 cores.
        # print 'Creating %d consumers' % num_consumers
        consumers = [ Consumer(tasks, results)
                      for i in xrange(num_consumers) ]
        for w in consumers:
            w.start()

        num_jobs = len(topk)
        # Enqueue jobs
        print len(topk)
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

        # for nid in topk:
        #     route = dij(graph, pw, nid)
        #     shortpaths[nid] = route
        #     # print route
        #     # if glb_shortest_paths.has_key(nid):
        #     #     shortpaths[nid] = route
        #     # else:
        #     #     route = dij(graph, pw, nid)
        #     #     shortpaths[nid] = route
        #     #     glb_shortest_paths[nid] = route
        # print 'finished'
        # return mset, sfscore

        pstar = None
        pdist = None
        for nid1 in topk: 
            for nid2 in topk:
                if nid2 != nid1:
                    dist = shortpaths[nid1][0][nid2]
                    if pdist == None or pdist > dist:
                        pstar = shortpaths[nid1][1][nid2]
                        pdist = dist

        if isinf(pdist):
            print '************ all infinite distances '
            break

#        print 'topk,', topk
        na = pstar[0]
        nb = pstar[-1]
#        print 'na, nb, dist', na, nb, pdist
#        print shortpaths[2][0]
        if len(topk) == 2:
            merge_set(graph, att, compdict, pstar, glb_shortest_paths)
            mset = recover_subgraph(compdict, pstar)
            flag = False
        else:
            [snid1, nc, sdist, spath] = topk_gain(shortpaths, topk, na, nb, pstar)
#            print shortpaths[2][0]
#            print 'pstar', pstar
#            print 'spath', spath
#            print 'snid1, na, nb, nc', snid1, na, nb, nc
            g = gain_path(list(set(pstar + spath)), att) - max(att[nc], gain_path(pstar, att))
            g = gain_path(list(set(pstar + spath)), att)
#            print att
#            print 'att[nc], gain_path(pstar, att)', att[nc], gain_path(pstar, att)
#            print 'list(set(pstar + spath))', list(set(pstar + spath))
#            for i in list(set(pstar + spath)):
#                print att[i]
#            print 'gain_path(list(set(pstar + spath)), att)', gain_path(list(set(pstar + spath)), att)
#            print 'g:', g
            if g < 0:
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
            if sfset == None:
                sfscore = f_score(mset, att, alpha)
                sfset = mset
            else:
                score = f_score(mset, att, alpha)
                if sfscore < score:
                    sfset = mset
                    sfscore = score
        iteration += 1
    return sfset, sfscore



# def printstat(graph, att, true_subgraph, subset_score, datasetname, alpha = 0.01):
# #    print subset_score, true_subgraph
#     if set(subset_score[0]) == set(true_subgraph):
#         print '--------------------------'
#         print '{0} (N = {1}, |S| = {2}): Passed'.format(datasetname, len(graph), len(true_subgraph))
# #        print 'true subgraph: {0}, {1}'.format(true_subgraph, f_score1(true_subgraph, att, alpha))
# #        print 'detected subgraph: {0}'.format(subset_score[0])
#         truescore = calc_true_subgraph_score(att, true_subgraph)
#         if truescore == subset_score[1]:
#             print 'true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1])
#         else:
#             print colored('true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1]), 'red')
#     else:
#         prec, recall = prec_recall(subset_score[0], true_subgraph)
#         print colored('##########################', 'red')
#         print colored('{0} (N = {1}, |S| = {2}): Failed'.format(datasetname, len(graph), len(true_subgraph)), 'red')
#         print colored('true subgraph length: {0}'.format(len(true_subgraph)), 'red')
#         print colored('detected subgraph length: {0}'.format(len(subset_score[0])), 'red')
#         print colored('precision: {0}, recall: {1}'.format(prec, recall), 'red')
# #        print colored('true subgraph score: {0}'.format(f_score1(true_subgraph, att, alpha)), 'red')
#         print colored('true subgraph: {0}, {1}'.format(true_subgraph, f_score1(true_subgraph, att, alpha)), 'red')
#         print colored('detected subgraph: {0}'.format(subset_score[0]), 'red')
#         print colored('true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1]), 'red')


def printstat(graph, att, true_subgraph, subset_score, datasetname, alpha = 0.01):
#    print subset_score, true_subgraph
#    print datasetname
    if set(subset_score[0]) == set(true_subgraph):
        prec,recall = prec_recall(subset_score[0], true_subgraph)
        print '--------------------------'
        print '{0} (N = {1}, |S| = {2}, N_alpha = {3}): Passed'.format(datasetname, len(graph), len(true_subgraph), len([item for item in att if item[0] <= 0.15]))
#        print 'true subgraph: {0}, {1}'.format(true_subgraph, f_score1(true_subgraph, att, alpha))
#        print 'detected subgraph: {0}'.format(subset_score[0])
        truescore = calc_true_subgraph_score(att, true_subgraph)
        if truescore == subset_score[1]:
            print 'true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1])
        else:
            print colored('true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1]), 'red')
    else:
        prec, recall = prec_recall(subset_score[0], true_subgraph)
        print colored('##########################', 'red')
        print colored('{0} (N = {1}, |S| = {2}, N_alpha = {3}): Failed'.format(datasetname, len(graph), len(true_subgraph), len([item for item in att if item[0] <= 0.15])), 'red')
        print colored('true subgraph length: {0}'.format(len(true_subgraph)), 'red')
        print colored('detected subgraph length: {0}'.format(len(subset_score[0])), 'red')
        print colored('precision: {0}, recall: {1}'.format(prec, recall), 'red')
#        print colored('true subgraph score: {0}'.format(f_score1(true_subgraph, att, alpha)), 'red')
        print colored('true subgraph: {0}, {1}'.format(true_subgraph, f_score1(true_subgraph, att, alpha)), 'red')
        print colored('detected subgraph: {0}'.format(subset_score[0]), 'red')
        print colored('true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1]), 'red')
    return prec,recall
         
def unittest():

    # datasetname = 'test dataset 01'
    # graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    # att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # # print att[4]
    # att[2][0] = 0.001
    # att[9][0] = 0.001
    # # att[8][0] = 0.001
    # # att[10][0] = 0.001
    # # att[7][0] = 0.01
    # # true_subgraph = [0, 1, 2, 3, 4, 5]
    # true_subgraph = [2, 5, 8, 9, 10, 11]
    # # print 'att', att
    # subset_score = additive_graphscan(graph, att)
    # # print subset_score
    # printstat(graph, att, true_subgraph, subset_score, datasetname)
    #
    # return

    datasetname = 'test dataset 00'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7], 7: [4, 6, 8], 8: [5, 7]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[6][0] = 0.01
    att[8][0] = 0.01
    att[2][0] = 0.01
    att[7][0] = 0.01
    # att[7][0] = 0.01
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [2, 5, 6, 7, 8]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    # return

    datasetname = 'test dataset 01'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[2][0] = 0.001
    att[6][0] = 0.001
    att[8][0] = 0.001
    att[10][0] = 0.001
    # att[7][0] = 0.01
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [2, 5, 6, 7, 8, 10]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    # print subset_score
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 02'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[2][0] = 0.001
    att[3][0] = 0.001
    att[8][0] = 0.001
    att[9][0] = 0.001
    # att[7][0] = 0.01
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [2, 3, 4, 5, 6, 8, 9]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

#    return 
    datasetname = 'test dataset 03'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[2][0] = 0.001
    att[0][0] = 0.001
    att[8][0] = 0.001
    att[9][0] = 0.001
    # att[7][0] = 0.01
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 1, 2, 5, 8, 9, 10, 11]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)


    datasetname = 'test dataset 04'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[0][0] = 0.001
    att[4][0] = 0.001
    att[8][0] = 0.001
    att[10][0] = 0.001
    # att[7][0] = 0.01
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 1, 4, 7, 8, 10]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 05'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[0][0] = 0.001
    att[1][0] = 0.001
    att[2][0] = 0.001
    att[9][0] = 0.001
    att[10][0] = 0.001
    att[11][0] = 0.001
    # att[7][0] = 0.01
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 1, 2, 5, 8, 9, 10, 11]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 06'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[0][0] = 0.001
    att[11][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 1, 2, 5, 8, 11]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 07'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[0][0] = 0.001
    att[11][0] = 0.001
    att[10][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 3, 6, 9, 10, 11]
    true_subgraph = [0, 1, 4, 7, 10, 11]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 08'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[0][0] = 0.001
    att[1][0] = 0.001
    att[10][0] = 0.001
    att[11][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 1, 4, 7, 10, 11]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 09'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[0][0] = 0.001
    att[3][0] = 0.001
    att[5][0] = 0.001
    att[8][0] = 0.001
    att[9][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 3, 4, 5, 6, 8, 9]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 10'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[0][0] = 0.001
    att[2][0] = 0.001
    att[9][0] = 0.001
    att[11][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [0, 1, 2]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    # return

    datasetname = 'test dataset 11'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[1][0] = 0.001
    att[10][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [1, 4, 7, 10]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 12'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[6][0] = 0.001
    att[8][0] = 0.001
    att[3][0] = 0.001
    att[2][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [2, 3, 5, 6, 7, 8]
    # print 'att', att
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)


    datasetname = 'test dataset 12'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 13'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.01], [0.01], [0.16], [0.01], [0.16], [0.01]]
    true_subgraph = [0, 1, 3, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 14'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.16], [0.01], [0.16], [0.01]]
    true_subgraph = [1, 3, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 15'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.16], [0.01], [0.16], [0.16]]
    true_subgraph = [0, 1, 3]
    subset_score = additive_graphscan(graph, att)
    print subset_score
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 16'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.01], [0.01], [0.16], [0.01], [0.16], [0.16]]
    true_subgraph = [0, 1, 3]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 17'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.01], [0.01], [0.01], [0.01]]
    true_subgraph = [1, 2, 3, 4, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 18'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.01], [0.16], [0.16], [0.16]]
    true_subgraph = [0, 1, 2]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 19'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 20'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.01], [0.01], [0.16], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 21'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.01], [0.16], [0.16], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
    datasetname = 'test dataset 22'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.03], [0.16], [0.03], [0.16], [0.16], [0.03]]
    true_subgraph = [0, 1, 2]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.03)
 
    datasetname = 'test dataset 23'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.00001], [0.16], [0.03], [0.16], [0.16], [0.03]]
    true_subgraph = [0]
#    f_score([0, ])(subset, att, compdict = {}, alpha = 0.05):
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.00001)
 
    datasetname = 'test dataset 24'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.05], [0.16], [0.16], [0.16], [0.16], [0.05]]
    true_subgraph = [0]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.05)
 
    datasetname = 'test dataset 25'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.16], [0.16], [0.16], [0.16], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.01)
 
    datasetname = 'test dataset 26'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.16], [0.16], [0.16], [0.16], [0.00001]]
    true_subgraph = [5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.00001)
 
    datasetname = 'test dataset 27'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.16], [0.01], [0.16], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = additive_graphscan(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)
 
 
def test_traffic_data():
    folders = ['waterData-noise-0', 'GridData', 'GridData1', 'GridData2', 'GridData3']
    for folder in folders[:1]:
        filenames = os.listdir(folder)
        filenames = [filename for filename in filenames if filename.find('readme') < 0]
        for filename in sorted(filenames, key = lambda item: len(item))[6:]:
            print os.path.join(folder, filename)
            start_time = time.time()
            graph, att, true_subgraph = read_traffic_data(os.path.join(folder, filename))
            # print 'true_subgraph', true_subgraph
            # print [att[i] for i in true_subgraph]
            if len(att) > 20000:
                continue
            subset_score = additive_graphscan(graph, att)
            print subset_score
            printstat(graph, att, true_subgraph, subset_score, filename)
            print("--- %s seconds ---" % (time.time() - start_time))
 
# def read_traffic_data(path):
#     att = []
#     lines = open(path).readlines()
#     n = -1
#     for idx, line in enumerate(lines):
#         if line.strip() == 'NodeID Weight':
#             n = idx + 1
#             break
#     for idx in range(n, len(lines)):
#         line = lines[idx]
#         if line.find('END') >= 0:
#             n = idx + 4
#             break
#         else:
#             items = line.split(' ')
#             att.append([float(items[1])])
#     graph = {}
#     for idx in range(n, len(lines)):
#         line = lines[idx]
#         if line.find('END') >= 0:
#             n = idx + 4
#             break
#         else:
#             items = line.split(' ')
#             n1 = int(items[0])
#             n2 = int(items[1])
#             if graph.has_key(n1):
#                 graph[n1].append(n2)
#             else:
#                 graph[n1] = [n2]
#     true_subgraph = []
#     for idx in range(n, len(lines)):
#         line = lines[idx]
#         if line.find('END') >= 0:
#             break
#         else:
#             items = line.split(' ')
#             true_subgraph.append(int(items[0]))
#             true_subgraph.append(int(items[1]))
#     true_subgraph = sorted(list(set(true_subgraph)))
#     att = add_noise(att)
#     return graph, att, true_subgraph

def add_noise(att, rate = 0.10):
    # return att
    if rate > 0:
        n = len(att)
        p = int(n * rate)
        rnd = np.random.permutation(n)
        for i in rnd[:p]:
            att[i][0] = 0.14
    return att

def test_water_pollution_data():
    folderName = 'WaterData'
    folderName = 'waterData-noise-0'
    subFolderNames = os.listdir(folderName)
    subFolderNames = [fileName for fileName in subFolderNames if os.path.isdir(folderName+'/'+fileName) ]

    for eachSubFolder in subFolderNames:
#        print eachSubFolder
        for eachFileData in os.listdir(os.path.join(folderName,eachSubFolder))[:1]:
            print os.path.join(folderName,eachSubFolder), eachFileData
#            print eachFileData.split("_")[5].split(".")[0]
            f = open("additive_result.txt",'a')
            start_time = time.time()
            graph, att,true_subgraph = read_traffic_data(os.path.join(folderName,eachSubFolder,eachFileData))

#            print graph
#            print att
            print 'true_subgraph', true_subgraph
            print [att[i] for i in true_subgraph]
            subset_score = additive_graphscan(graph, att)
            prec,recall = printstat(graph,att,true_subgraph,subset_score,eachFileData)
            print eachFileData
            print prec,recall
            timeV = time.time() - start_time
            f.write(str(eachFileData.split("_")[5].split(".")[0]+" "+str(prec)+" "+str(recall)+" "+str(timeV)))
            print("--- %s seconds ---" % (time.time() - start_time))
            f.close()


def getOriginalNodes():
    users2ID = dict()
    with open('../../realDataSet/hazeData/users2ID.txt') as f:
        for eachLine in f.readlines():
            items = eachLine.rstrip().split(' ')
            id = items[0]
            changed = items[1]
            users2ID[int(changed)] = int(id)
    return users2ID


def read_APDM_data(path):
    graph = {}
    att = []
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
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            n1 = int(items[0])
            n2 = int(items[1])
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
            if items[0].startswith('null'): break
            true_subgraph.append(int(items[0]))
            true_subgraph.append(int(items[1]))
    true_subgraph = sorted(list(set(true_subgraph)))
    return graph, att, true_subgraph


def read_traffic_data(path):
    att = []
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
    att = add_noise(att)

    return graph, att, true_subgraph

def test():
    graph = {0: [], 1: [], 2: [3, 4, 6], 3: [2, 5, 6], 4: [2, 6, 6], 5: [3, 6, 6], 6: [2, 3, 4, 5]}
    graph1 = dict()
    for nid, neis in graph.items():
        if len(neis) > 0:
            graph1[nid] = neis
    pw = [0, 0, 0, 0, 0, 0, 0]
    src = 2
    route = dij(graph1, pw, src)
    printstat1(graph1, pw, src, route)

def finalTest_AdditiveScan_multiwCore_WaterNetwork():
    resultGraphFolder = '../../ComparePreRecROC/WaterNetwork_12500/AdditiveScan/graphResults/'
    resultFileName = '../../ComparePreRecROC/WaterNetwork_12500/AdditiveScan/waterNetworkResult_AdditiveScan.txt'
    rootFolder = '../../realDataSet/WaterData/source_12500'
    controlFileList = '../../ComparePreRecROC/WaterNetwork_12500/controlFileList.txt'
    resultFileName = 'additiveScan_waterNetwork.txt'
    iterations_bound = 10
    ncores = 1
    minutes = 10000
    files = []
    with open('../../ComparePreRecROC/WaterNetwork_12500/controlFileList.txt') as f:
        for eachLine in f.readlines():
            if eachLine.rstrip().endswith('.txt'):
                files.append(eachLine.rstrip()) 
    for eachFile in files:
        print 'current processing file : ', os.path.join(eachFile)
        date = eachFile.split('.txt')[0]
        eachFileCase = os.path.join(rootFolder,eachFile)
        start_time = time.time()
        graph, att, trueSubGraph = read_APDM_data(eachFileCase)
        subset_score = additive_graphscan(graph, att,ncores,iterations_bound,minutes)
        resultNodes = subset_score[0]
        bjscore = subset_score[1]
        runningTime = time.time() - start_time
        file = open(resultFileName,'a')
        
        resultSet = set(resultNodes)
        trueSubGraph = set(trueSubGraph)
        intersect = resultSet.intersection(trueSubGraph)
        pre = (len(intersect)*1.0) / (len(resultNodes)*1.0)
        rec = (len(intersect)*1.0) / (len(trueSubGraph)*1.0)
        if pre+rec == 0.0:
            fmeasure = 0.0
        else:
            fmeasure = 2.0*(pre*rec) / (pre+rec)
            
        truePositive = len(intersect)*1.0 ;
        falseNegative = len(trueSubGraph)*1.0 - truePositive ;
        falsePositive = len(resultNodes)*1.0 - len(intersect)*1.0 ;
        trueNegative = len(att) - len(trueSubGraph) - falsePositive ;
        tpr = truePositive / (truePositive+falseNegative) ;
        fpr = trueNegative / (falsePositive + trueNegative) ;    
        
        file.write(
                   "{0:.6f}".format(0.0)+"\t"+
                   "{0:.6f}".format(bjscore)+"\t"+
                   "{0:.6f}".format(runningTime)+"\t"+
                   "{0:06d}".format(1000)+"\t"+
                   "{0:06d}".format(len(subset_score[0]))+"\t"+
                   date+"\t"+
                   "{0:.6f}".format(0.0)+"\t"+
                   "{0:.6f}".format(pre)+"\t"+
                   "{0:.6f}".format(rec)+"\t"+
                   "{0:.6f}".format(fmeasure)+"\t"+
                   "{0:.6f}".format(tpr)+"\t"+
                   "{0:.6f}".format(fpr)+"\n")
        file.close()
        try: os.stat(resultGraphFolder)
        except: os.mkdir(resultGraphFolder)
        f = open(os.path.join(resultGraphFolder,eachFile),'w')
        if len(subset_score[0]) == 0:
            f.write('null')
            f.close()
        else:
            for node in subset_score[0]:
                id = node
                f.write(str(id)+'\n')
            f.close()

if __name__ == "__main__":
    finalTest_AdditiveScan_multiwCore_WaterNetwork()
