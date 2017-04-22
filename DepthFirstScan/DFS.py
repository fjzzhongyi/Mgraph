__author__ = 'fengchen'
import numpy as np
import copy
import os
import time,sys,datetime
from pyspark import SparkContext, SparkConf

# Queue simulates a queue
class Queue:
    def __init__(self, item = None):
        self.items = list()
        if item:
            self.items.append(item)
        pass

    def len(self):
        return len(self.items)

    def push(self, item):
        # self.items.append(item)
        self.items.insert(0, item)
        self.items = sorted(self.items, key = lambda item: item[1])

    def pop(self):
        return self.items.pop()


def getcont(nid, att, compdict, alpha):
    # get the size of nodes comp contains
    if compdict.has_key(nid):
        return len(compdict[nid])
    else:
        if att[nid][0] <= alpha:
            return 1
        else:
            return 0


# what is Route
class Route:
    def __init__(self, s = None, graph = None, att = None, nodepri = None, compdict = None, alpha = None):
        if graph:
            self.nodepri = nodepri
            n = len(graph)
            self.N = n
            self.graph = graph
            self.att = att
            self.incl = dict()
            self.incl[s] = 1
            self.excl = dict()
            # self.bin = [2] * n
            # self.bin[s] = 1
            self.path = [s]
            self.priority = setprioity(self.path, att)
            self.sidetracks = []
            self.sin = []
            self.sins = 0
            self.souts = 0
            self.sout = []
            self.compdict = compdict
            self.alpha = alpha
            # self.nodepri =
        else:
            self.nodepri = nodepri
            self.N = 0
            self.incl = dict()
            self.excl = dict()
            # self.bin = []
            self.path = []
            self.sidetracks = []
            self.sin = []
            self.sout = []
            self.priority = 0
            self.graph = None
            self.att = None
            self.sins = 0
            self.souts = 0
            self.compdict = {}
            self.alpha = 0.05


    def get_current_location(self):
        return self.path[-1]


    def getsubset(self):
        return self.incl.keys()


    def get_neis(self):
        neis = self.graph[self.get_current_location()]
        res = []
        for nei in neis:
            if not self.incl.has_key(nei) and not self.excl.has_key(nei):
                if not set(self.graph[nei]).intersection(self.path[:-1]):
                    res.append(nei)
        return sorted(res, key=lambda item: self.nodepri[item] * -1)


    def getsout(self, nid0):
        sout = []
        souts = 0.0
        Q = []
        if self.nodepri[nid0] > 0 and self.excl.has_key(nid0):
            Q.append(nid0)
        while Q:
            nid = Q.pop()
            sout.append(nid)
            souts += self.nodepri[nid]
            for nei in [i for i in self.graph[nid] if not self.incl.has_key(i)]:
                if self.nodepri[nei] > 0 and nei not in Q and nei not in sout:
                    Q.append(nei)
            if len(sout) > 5:
                break
        return sout, souts


    def expand_path(self, idx, neis):
        nid = neis[idx]
        self.path.append(nid)
        self.incl[nid] = 1
        for nei in neis[:idx]:
            self.excl[nei] = 1
        for nei in neis[:idx]:
            sout, souts = self.getsout(nei)
            if souts > self.souts:
                self.souts = souts

# what is Route
class Route:
    def __init__(self, s = None, graph = None, att = None, nodepri = None, compdict = None, alpha = None):
        if graph:
            self.nodepri = nodepri
            n = len(graph)
            self.N = n
            self.graph = graph
            self.att = att
            
            # what's incl & excl:  include & exclude
            self.incl = dict()
            self.incl[s] = 1
            self.excl = dict()
            
            # self.bin = [2] * n
            # self.bin[s] = 1
            # path starts from s
            self.path = [s]
            self.priority = setprioity(self.path, att)
            self.sidetracks = []
            
            self.sin = []
            self.sins = 0
            self.souts = 0
            self.sout = []
            
            self.compdict = compdict
            self.alpha = alpha
            # self.nodepri =
        else:
            self.nodepri = nodepri
            self.N = 0
            self.incl = dict()
            self.excl = dict()
            # self.bin = []
            self.path = []
            self.sidetracks = []
            self.sin = []
            self.sout = []
            self.priority = 0
            self.graph = None
            self.att = None
            self.sins = 0
            self.souts = 0
            self.compdict = {}
            self.alpha = 0.05


    def get_current_location(self):
        return self.path[-1]


    def getsubset(self):
        return self.incl.keys()


    def get_neis(self):
        # this func returns the candidate nodes(not only anomalous) that may be included in next step
        # get res from neis.  res is subset of neis, excluding those 
        neis = self.graph[self.get_current_location()]
        res = []
        for nei in neis:
            # not to form a loop
            if not self.incl.has_key(nei) and not self.excl.has_key(nei):
                if not set(self.graph[nei]).intersection(self.path[:-1]):
                    res.append(nei)
        # the bigger the nodepri is, the priorer ...
        return sorted(res, key=lambda item: self.nodepri[item] * -1)


    def getsout(self, nid0):
        # the func 
        sout = []
        souts = 0.0
        Q = []
        if self.nodepri[nid0] > 0 and self.excl.has_key(nid0):
            Q.append(nid0)
        while Q:
            nid = Q.pop()
            sout.append(nid)
            souts += self.nodepri[nid]
            for nei in [i for i in self.graph[nid] if not self.incl.has_key(i)]:
                if self.nodepri[nei] > 0 and nei not in Q and nei not in sout:
                    Q.append(nei)
            if len(sout) > 5:
                break
        return sout, souts


    def expand_path(self, idx, neis):
        # how to expand the path 
        nid = neis[idx]
        self.path.append(nid)
        # route includes neis[idx], and excludes neis[:idex]
        self.incl[nid] = 1
        for nei in neis[:idx]:
            self.excl[nei] = 1
        for nei in neis[:idx]:
            sout, souts = self.getsout(nei)
            # keep the biggest souts
            if souts > self.souts:
                self.souts = souts


    def backtrack(self, neis):
        nid = self.path.pop()

        for nei in neis:
            self.excl[nei] = 1

        for nei in neis:
            sout, souts = self.getsout(nei)
            if souts > self.souts:
                self.souts = souts

        b = False
        for st in self.sidetracks:
            if nid in self.graph[st[-1]]:
                st.append(nid)
                b = True
                sins = setprioity(st, self.att, self.compdict, self.alpha)
                if sins > self.sins:
                    self.sins = sins
                if sins <= 0:
                    return False
                break
        if not b:
            if self.nodepri[nid] >= 1:
                st = [nid]
                sins = setprioity(st, self.att, self.compdict, self.alpha)
                self.sidetracks.append(st)
                if sins > self.sins:
                    self.sins = sins
            else:
                self.excl[nid] = 1
                del self.incl[nid]
                return False
        return True


def kl(ratio, alpha):
    if ratio == 0:
        ratio += 0.0000000001
    elif ratio == 1:
        ratio -= 0.0000000001
    alpha += 0.0000000001
    return ratio * np.log(ratio / alpha) + (1 - ratio) * np.log((1 - ratio) / (1 - alpha))


# Each node has a single attribute p-value
def f_score(subset, att, compdict = {}, alpha = 0.05):
    n = 0
    nalpha = 0
    for nid in subset:
        if att[nid][0] <= alpha:
            c = getcont(nid, att, compdict, alpha)
            nalpha += c
            n += c
        else:
            n += 1
#    nalpha = len([att[nid][0] for nid in subset if att[nid][0] <= alpha])
    return n * kl(nalpha / n, alpha)


def f_score1(subset, att, alpha = 0.05):
    n = len(subset) * 1.0
    nalpha = len([att[nid][0] for nid in subset if att[nid][0] <= alpha])
    return n * kl(nalpha / n, alpha)


# we only conisder one attribut (p-value)
def nodepriority(nid, att, compdict = {}, alpha = 0.05):
    if att[nid][0] <= alpha:
        return getcont(nid, att, compdict, alpha)
    else:
        return 0
    # return -1 * att[nid][0]

# this func return the total number of input subset's abnormal nodes
def setprioity(subset, att, compdict = {}, alpha=0.05):
    """ 
    n = 0
    for i in subset:
        if att[i][0] <= alpha:
            n += getcont(i, att, compdict, alpha)
        else:
            n -= 1
    """
    #SPARK
    if len(subset)<=50:
        n = 0
        for i in subset:
            if att[i][0] <= alpha:
                n += getcont(i, att, compdict, alpha)
            else:
                n -= 1
    else:
        def spark_getcont(nid):
            # get the size of nodes comp contains
            print alpha    
            if att[nid][0] <= alpha:
                if compdict.has_key(nid):
                    return len(compdict[nid])
                else:
                    if att[nid][0] <= alpha:
                        return 1
                    else:
                        return 0
            else:
                return -1
        sc=SparkContext()
        rdd=sc.parallelize(subset)
        sc.broadcast(att)
        sc.broadcast(compdict)
        sc.broadcast(alpha)
        n=rdd.map(spark_getcont).reduce(lambda a,b:a+b)
        sc.stop()
    
    return n

def refine_graph(graph1, att1, alpha):
    # return graph1, att1, {};  graph is a dict(), att1 is a list
    # how to refine the graph:
    # extract connected anomalous nodes into one virtual node
    graph = copy.deepcopy(graph1)
    att = copy.deepcopy(att1)
    bigcomp = []
    comps = []
    # search the whole graph 
    for nid, neis in graph.items():
            # start from node nid, recursively add all connected anomalous nodes(pvalue<=alpha), which composes a subgraph
            comp = [nid]
            queue = [nid]
            while queue:
                i = queue.pop()
                # list graph[i]'s edges 
                for j in graph[i]:
                    if att[j][0] <= alpha and j not in comp:
                        queue.append(j)
                        comp.append(j)
                        bigcomp.append(j)
            comps.append(comp)
    # filter comp whoes size is only one, which means comps'length may less than n
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
                    graph[j].append(n)
            # regrad i as normal
            graph[i] = []
            att[i][0] = 1
        # regard the subgraph as a new node
        graph[n] = list(set(neis))
        att.append([alpha])
        compdict[n] = comp
        n += 1
    # so now the lenght of graph extended from n to n~2n
    # compdict[n]: nodes that new virtual node n exactly contains.  NOTE that all nodes are abnormal
    return graph, att, compdict


def proc(route, priqueue, att, compdict, alpha, subset_score, anomaly_ratio, radius):
    if route.path and (route.souts < route.sins or route.souts == 0):
        np = 0.0
        nn = 0.0
        np1 = 0
        subset = []
        for nid in range(len(att)):
            if route.incl.has_key(nid):
                if route.nodepri[nid] >= 1:
                    np += route.nodepri[nid]
                else:
                    nn += 1
                subset.append(nid)
            elif not route.excl.has_key(nid) and route.nodepri[nid] >= 1:
                np1 += route.nodepri[nid]
        n = np + nn
        fs = n * kl(np / n, alpha)
        if not subset_score or subset_score[1] < fs:
            subset_score = [subset, fs, alpha]

        LBound = fs
        UBound = (n + np1) * kl((np + np1) / (n + np1), alpha)
        if subset_score[1] < LBound and LBound < UBound:
            subset_score = [subset, fs, alpha]
            if route.path and anomaly_rate(route, compdict) > anomaly_ratio and len(route.path) <= radius:
                priqueue.push([route, np])
        elif subset_score[1] < LBound and LBound == UBound:
            subset_score = [subset, fs, alpha]
        elif UBound <= subset_score[1]:
            pass
        elif LBound <= subset_score[1] and subset_score[1] < UBound:
            if route.path and anomaly_rate(route, compdict) > anomaly_ratio and len(route.path) <= radius:
                priqueue.push([route, np])
    else:
        pass
    return subset_score


def anomaly_rate(route, compdict):
    # this cals out anomaly nodes/ all nodes
    n1 = 0.0
    n2 = 0.0
    for nid in route.incl.keys():
        if route.att[nid][0] <= route.alpha:
            if compdict.has_key(nid):
                n1 += len(compdict[nid])
            else:
                n1 += 1
        else:
            n2 += 1
    return n1 / (n1 + n2)
    
    
def get_seeds(graph, att, compdict, alpha):
    # print graph, att
    seeds = []
    for nid, neis in graph.items():
        if nodepriority(nid, att, compdict, alpha) > 1:
            seeds.append([nid, nodepriority(nid, att, compdict, alpha)])
        elif nodepriority(nid, att, compdict, alpha) == 1 and min([nodepriority(nei, att, compdict, alpha) for nei in neis]) == 1:
            seeds.append([nid, nodepriority(nid, att, compdict, alpha)])
    if not seeds:
        for nid, neis in graph.items():
            if nodepriority(nid, att, compdict, alpha) >= 1 and min([nodepriority(nei, att, compdict, alpha) for nei in neis]) == 0:
                seeds.append([nid, nodepriority(nid, att, compdict, alpha)])
    seeds = sorted(seeds, key=lambda item: item[1] * -1)
    # guess seeds are set of nodes (from which we continue our depth first scan in sequence). Other words, we should scan from first of seeds
    return seeds

    
# graph: {node_id: [node_ids]}. att: [[],[],[]]
# Identify seeds nodes that have higher priorities than their neighbors
def depth_first_subgraph_detection(graph, att, radius = 7, anomaly_ratio = 0.5, minutes = 30, alpha_max = 0.15):
    
    # note: the func need parameter alpha_max, not alpha
    if not radius:
        radius = len(graph)
    start_time = time.time()
    subset_score = None
    # those anomalous p-values and sort them from least to most anomalous
    alphas = set(item[0] for item in att if item[0] != 0 and item[0] < alpha_max)
    alphas = sorted(list(alphas), key = lambda item: item * -1)
    ori_graph = graph
    ori_att = att
    for alpha_i, alpha in enumerate(alphas):
        print alpha_i, alpha
        graph, att, compdict = refine_graph(ori_graph, ori_att, alpha)
        # for each node in graph, calculate its priority.
        # the larger its value is,the priorer it is
        nodeprio = [0] * len(graph)
        for nid in range(len(graph)):
            nodeprio[nid] = nodepriority(nid, att, compdict, alpha)
        seeds = get_seeds(graph, att, compdict, alpha)
        # print 'seeds', seeds
        print compdict

        # only search first 5 (at most) of the seeds 
        for seed_i, seed in enumerate(seeds[:5]):
            if not subset_score or seed[0] not in subset_score[0] or seed_i == 0:
                # print 'compdict[seed[0]]', compdict[seed[0]]
                route = Route(seed[0], graph, att, nodeprio, compdict, alpha)
                print seed
                # why its priority is 1?
                priqueue = Queue([route, 1])
                if not subset_score:
                    subset_score = [route.getsubset(), f_score(route.getsubset(), att, compdict), alpha]
                iter = 0
                while priqueue.len() > 0:
                    [route, pri] = priqueue.pop()
                    # print 'pop:', route.path, route.incl.keys(), subset_score
                    iter += 1
                    neis = route.get_neis()
                    if iter % 5000 == 0:
                        print 'iterations: ', iter
                        print 'pop:', route.path, route.incl.keys()
                        duration = (time.time() - start_time)
                        # if duration reached out of time, return none immediately
                        if duration > minutes * 60:
                            print 'time limit reached... '
                            oriset = []
                            if subset_score:
                                for i in subset_score[0]:
                                    if compdict.has_key(i):
                                        oriset.extend(compdict[i])
                                    else:
                                        oriset.append(i)
                                subset_score[0] = sorted(oriset)
                            else:
                                subset_score = [[], 0]
                            return subset_score
                    # print iter, neis
                    # calc new routes
                    for idx, nei in enumerate(neis):
                        newroute = Route(seed[0], graph, att, nodeprio, compdict, alpha)
                        newroute.incl = copy.deepcopy(route.incl)
                        newroute.excl = copy.deepcopy(route.excl)
                        newroute.path = copy.deepcopy(route.path)
                        newroute.sidetracks = copy.deepcopy(route.sidetracks)
                        newroute.souts = route.souts
                        newroute.sins  = route.sins
                        newroute.expand_path(idx, neis)
                        # print 'newroute.path', newroute.path, newroute.souts, newroute.sins
                        subset_score = proc(newroute, priqueue, att, compdict, alpha, subset_score, anomaly_ratio, radius)
                    re = route.backtrack(neis)
                    if re:
                        subset_score = proc(route, priqueue, att, compdict, alpha, subset_score, anomaly_ratio, radius)
    oriset = []
    if subset_score:
        for i in subset_score[0]:
            if compdict.has_key(i):
                oriset.extend(compdict[i])
            else:
                oriset.append(i)
        subset_score[0] = sorted(oriset)
    else:
        subset_score = [[], 0]
    print subset_score

    return subset_score


def prec_recall(detect_subgraph, true_subgraph):
    n = 0.0
#    print detect_subgraph
    for i in detect_subgraph:
        if i in true_subgraph:
            n += 1
    if detect_subgraph:
        prec = n / len(detect_subgraph)
    else:
        prec = 0
    recall = n / len(true_subgraph)
    return prec, recall
    

#def printstat(graph, att, true_subgraph, subset_score, datasetname, alpha = 0.01):
#    if set(subset_score[0]) == set(true_subgraph):
#        print '--------------------------'
#        print '{0} (N = {1}, |S| = {2}): Passed'.format(datasetname, len(graph), len(true_subgraph))
##        print 'true subgraph: {0}, {1}'.format(true_subgraph, f_score1(true_subgraph, att, alpha))
##        print 'detected subgraph: {0}'.format(subset_score[0])
#        truescore = calc_true_subgraph_score(att, true_subgraph)
#        if truescore == subset_score[1]:
#            print 'true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1])
#        else:
#            print colored('true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1]), 'red')
#            
#    else:
#        prec, recall = prec_recall(subset_score[0], true_subgraph)
#        print colored('##########################', 'red')
#        print colored('{0} (N = {1}, |S| = {2}): Failed'.format(datasetname, len(graph), len(true_subgraph)), 'red')
#        print colored('true subgraph length: {0}'.format(len(true_subgraph)), 'red')
#        print colored('detected subgraph length: {0}'.format(len(subset_score[0])), 'red')
#        print colored('precision: {0}, recall: {1}'.format(prec, recall), 'red')
##        print colored('true subgraph score: {0}'.format(f_score1(true_subgraph, att, alpha)), 'red')
##        print colored('true subgraph: {0}, {1}'.format(true_subgraph, f_score1(true_subgraph, att, alpha)), 'red')
##        print colored('detected subgraph: {0}'.format(subset_score[0]), 'red')
#        print colored('true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1]), 'red')

  
def printstat(graph, att, true_subgraph, subset_score, datasetname, alpha = 0.01):
#    print subset_score, true_subgraph
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
            print 'true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1])
    else:
        prec, recall = prec_recall(subset_score[0], true_subgraph)
        truescore = subset_score[1]
        print '##########################'
        print '{0} (N = {1}, |S| = {2}, N_alpha = {3}): Failed'.format(datasetname, len(graph), len(true_subgraph), len([item for item in att if item[0] <= 0.15]))
        print 'true subgraph length: {0}'.format(len(true_subgraph))
        print 'detected subgraph length: {0}'.format(len(subset_score[0]))
        print 'precision: {0}, recall: {1}'.format(prec, recall)
#        print 'true subgraph score: {0}'.format(f_score1(true_subgraph, att, alpha))
        print 'true subgraph: {0}, {1}'.format(true_subgraph, f_score1(true_subgraph, att, alpha))
        print 'detected subgraph: {0}'.format(subset_score[0])
        print 'true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1])
    return prec,recall,truescore


def calc_true_subgraph_score(att, true_subgraph):
    alphas = list(set([att[i][0] for i in true_subgraph]))
    score = None
    for alpha in alphas:
        s = f_score1(true_subgraph, att, alpha)
        if not score or s > score:
            score = s
    return s


def cond(x, a, b):
    if x >= a and x <= b:
        return True
    else:
        return False


def unittest():

    for iter in range(0):
        M = 90
        N = M * M

        graph = {}
        for row in range(M):
            for column in range(M):
                n = row * M + column
                graph[n] = []
                if column > 0:
                    graph[n].append(n-1)

                if column < M-1:
                    graph[n].append(n+1)

                if row > 0:
                    graph[n].append((row - 1) * M + column)

                if row < M - 1:
                    graph[n].append((row + 1) * M + column)

        att = []
        for i in range(N):
            att.append([0.61])
        # row_col = [[1, 2], [2, 3], [3, 4], [4, 5]]
        # row_col = [[1, 2], [1, 3], [2, 2], [2, 3], [5, 2], [5, 3], [6, 2], [6, 3]]
        # row_col = [[1, 2], [1, 3], [2, 2], [2, 3], [8, 2], [8, 3], [9, 2], [9, 3]]
        # row_col = [[0, 2], [0, 3], [9, 2], [9, 3]]
        row_col = []
        for i in range(20):
            row_col.append([5, i])
            row_col.append([7, i])
        # row_col = [[5, 2], [5, 3]]
        true_subgraph = []
        for row, col in row_col:
            n = row * M + col
            # print row, col, n
            att[n][0] = 0.01
            true_subgraph.append(n)
        # print att
        # print graph
        print 'true subgraph', true_subgraph
        att = add_noise(att, 0.10)
        datasetname = 'simuation'
        start_time = time.time()
        print 'start detection'
        subset_score = depth_first_subgraph_detection(graph, att)
        print subset_score
        printstat(graph, att, true_subgraph, subset_score, datasetname)
        print("--- %s seconds ---" % (time.time() - start_time))


    # return


    # datasetname = 'test dataset 1'
    # graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    # att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # # print att[4]
    # att[2][0] = 0.001
    # att[0][0] = 0.001
    # att[8][0] = 0.001
    # att[9][0] = 0.001
    # # att[7][0] = 0.01
    # # true_subgraph = [0, 1, 2, 3, 4, 5]
    # true_subgraph = [0, 1, 2, 5, 6, 7, 8, 9]
    # print 'att', att
    # subset_score = depth_first_subgraph_detection(graph, att)
    # printstat(graph, att, true_subgraph, subset_score, datasetname)
    #
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
    subset_score = depth_first_subgraph_detection(graph, att)
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
    subset_score = depth_first_subgraph_detection(graph, att)
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
    true_subgraph = [2, 3, 5, 6, 7, 8, 9]
    # print 'att', att
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

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
    true_subgraph = [0, 1, 2, 5, 6, 7, 8, 9]
    # print 'att', att
    subset_score = depth_first_subgraph_detection(graph, att)
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
    subset_score = depth_first_subgraph_detection(graph, att)
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
    true_subgraph = [0, 1, 2, 3, 6, 9, 10, 11]
    # print 'att', att
    subset_score = depth_first_subgraph_detection(graph, att)
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
    subset_score = depth_first_subgraph_detection(graph, att)
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
    # print 'att', att
    subset_score = depth_first_subgraph_detection(graph, att)
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
    subset_score = depth_first_subgraph_detection(graph, att)
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
    subset_score = depth_first_subgraph_detection(graph, att)
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
    true_subgraph = [0, 1, 2, 5, 8, 9, 10, 11]
    # print 'att', att
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

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
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 12'
    graph = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8], 6: [3, 7, 9], 7: [4, 6, 8, 10], 8: [5, 7, 11], 9: [6, 10], 10: [7, 9, 11], 11: [8, 10]}
    # att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    # att = [[0.01], [0.61], [0.61], [0.61], [0.61], [0.01]]
    att = [[0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61], [0.61]]
    # print att[4]
    att[6][0] = 0.001
    att[8][0] = 0.001
    att[2][0] = 0.001
    # true_subgraph = [0, 1, 2, 3, 4, 5]
    true_subgraph = [2, 5, 6, 7, 8]
    # print 'att', att
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    return


    datasetname = 'test dataset 1'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 2'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.01], [0.01], [0.16], [0.01], [0.16], [0.01]]
    true_subgraph = [0, 1, 3, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 3'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.16], [0.01], [0.16], [0.01]]
    true_subgraph = [1, 3, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 4'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.16], [0.01], [0.16], [0.16]]
    true_subgraph = [1, 3, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 5'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.01], [0.01], [0.16], [0.01], [0.16], [0.16]]
    true_subgraph = [0, 1, 3]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 6'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.01], [0.01], [0.01], [0.01]]
    true_subgraph = [1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 7'
    graph = {0: [1, 2, 3, 4, 5], 1: [0, 4, 5], 2: [0, 3, 4], 3: [0, 2, 5], 4: [0, 1, 2], 5: [0, 1, 3]}
    att = [[0.16], [0.01], [0.01], [0.16], [0.16], [0.16]]
    true_subgraph = [1, 2, 4]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 8'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.01], [0.01], [0.01], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 9'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.01], [0.01], [0.16], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 10'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.01], [0.16], [0.16], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

    datasetname = 'test dataset 11'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.03], [0.16], [0.03], [0.16], [0.16], [0.03]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.03)

    datasetname = 'test dataset 12'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.00001], [0.16], [0.03], [0.16], [0.16], [0.03]]
    true_subgraph = [0]
#    f_score([0, ])(subset, att, compdict = {}, alpha = 0.05):
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.00001)

    datasetname = 'test dataset 13'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.05], [0.16], [0.16], [0.16], [0.16], [0.05]]
    true_subgraph = [0]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.05)

    datasetname = 'test dataset 14'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.16], [0.16], [0.16], [0.16], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.01)

    datasetname = 'test dataset 15'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.16], [0.16], [0.16], [0.16], [0.00001]]
    true_subgraph = [5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname, 0.00001)

    datasetname = 'test dataset 16'
    graph = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
    att = [[0.01], [0.16], [0.01], [0.16], [0.01], [0.01]]
    true_subgraph = [0, 1, 2, 3, 4, 5]
    subset_score = depth_first_subgraph_detection(graph, att)
    printstat(graph, att, true_subgraph, subset_score, datasetname)

def test_graph(graph):
    for nid, neis in graph.items():
        for nei in neis:
            if nid not in graph[nei]:
                return False
    return True

# def test_traffic_data():
#     folders = ['GridData', 'GridData1', 'GridData2', 'GridData3']
#     #folders = ['C:\\Users\\Feng Chen\\Dropbox\\WaterData']
#     for folder in folders[:]:
#         filenames = os.listdir(folder)
#         filenames = [filename for filename in filenames if filename.find('readme') < 0]
#         for filename in sorted(filenames, key = lambda item: len(item))[:]:
#             print os.path.join(folder, filename)
#             start_time = time.time()
#             graph, att, true_subgraph = read_traffic_data(os.path.join(folder, filename))
#             if not test_graph(graph):
#                 print '********************'
#             subset_score = depth_first_subgraph_detection(graph, att)
#             printstat(graph, att, true_subgraph, subset_score, filename)
#             print("--- %s seconds ---" % (time.time() - start_time))


def test_traffic_data():
    folders = ['waterData-noise-0', 'GridData', 'GridData1', 'GridData2', 'GridData3']
    for folder in folders[:1]:
        filenames = os.listdir(folder)
        filenames = [filename for filename in filenames if filename.find('readme') < 0]
        for filename in sorted(filenames, key = lambda item: len(item))[:]:
            print os.path.join(folder, filename)
            start_time = time.time()
            graph, att, true_subgraph = read_traffic_data(os.path.join(folder, filename))
            # print 'true_subgraph', true_subgraph
            # print [att[i] for i in true_subgraph]
            if len(att) > 20000:
                continue
            subset_score = depth_first_subgraph_detection(graph, att)
            print subset_score
            printstat(graph, att, true_subgraph, subset_score, filename)
            print("--- %s seconds ---" % (time.time() - start_time))


def test_water_pollution_data():
    folderName = 'WaterData'
    subFolderNames = os.listdir(folderName)
    subFolderNames = [fileName for fileName in subFolderNames if os.path.isdir(folderName+'/'+fileName) ]
    
    for eachSubFolder in subFolderNames:
#        print eachSubFolder
#        print os.path.join(folderName,eachSubFolder)
        for eachFileData in os.listdir(os.path.join(folderName,eachSubFolder)):
#            print eachFileData.split("_")[5].split(".")[0]
            f = open("additive_result.txt",'a')
            start_time = time.time()
            graph, att,true_subgraph = read_traffic_data(os.path.join(folderName,eachSubFolder,eachFileData))
#            print graph
#            print att
#            print true_subgraph
            subset_score = depth_first_subgraph_detection(graph, att)
            prec,recall = printstat(graph,att,true_subgraph,subset_score,eachFileData)
            print eachFileData
            print prec,recall
            timeV = time.time() - start_time
            f.write(str(eachFileData.split("_")[5].split(".")[0]+" "+str(prec)+" "+str(recall)+" "+str(timeV)))
            print("--- %s seconds ---" % (time.time() - start_time))
            f.close()


def add_noise(att, rate = 0.10):
    # return att
    if rate > 0:
        n = len(att)
        p = int(n * rate)
        rnd = np.random.permutation(n)
        for i in rnd[:p]:
            att[i][0] = 0.14
    return att


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
    
# def add_noise(att, rate = 0.20):
#     # return att
#     if rate > 0:
#         n = len(att)
#         p = int(n * rate)
#         rnd = np.random.permutation(n)
#         for i in rnd[:p]:
#             att[i][0] = 0.05
#     return att

#def read_traffic_data(path):
#    att = []
#    lines = open(path).readlines()
#    n = -1
#    for idx, line in enumerate(lines):
#        if line.strip() == 'NodeID Weight':
#            n = idx + 1
#            break
#    for idx in range(n, len(lines)):
#        line = lines[idx]
#        if line.find('END') >= 0:
#            n = idx + 4
#            break
#        else:
#            items = line.split(' ')
#            att.append([float(items[1])])
#    graph = {}
#    for idx in range(n, len(lines)):
#        line = lines[idx]
#        if line.find('END') >= 0:
#            n = idx + 4
#            break
#        else:
#            items = line.split(' ')
#            n1 = int(items[0])
#            n2 = int(items[1])
#            if graph.has_key(n1):
#                graph[n1].append(n2)
#            else:
#                graph[n1] = [n2]
#    true_subgraph = []
#    for idx in range(n, len(lines)):
#        line = lines[idx]
#        if line.find('END') >= 0:
#            break
#        else:
#            items = line.split(' ')
#            true_subgraph.append(int(items[0]))
#            true_subgraph.append(int(items[1]))
#    true_subgraph = sorted(list(set(true_subgraph)))
#    return graph, att, true_subgraph


def test_water_pollution_data_noise_x(x,hours):
    folderName = '../static'
    subFolderNames = os.listdir(folderName)
    print subFolderNames
    subFolderNames = [fileName for fileName in subFolderNames if os.path.isdir(folderName+'/'+fileName) ]
    for eachSubFolder in subFolderNames:
        if eachSubFolder == 'source_12500':
            for eachFileData in os.listdir(os.path.join(folderName,eachSubFolder)):
                if str(eachFileData.split("_")[5].split(".")[0]) == x and eachFileData.split("_")[2] in hours:
                    file = open("depth1.8.txt",'a')
                    start_time = time.time()
                    print 'current processing file : ', os.path.join(folderName,eachSubFolder,eachFileData)
                    data_source = str(eachSubFolder)
                    fileName = str(eachFileData)
                    graph, att,true_subgraph = read_traffic_data(os.path.join(folderName,eachSubFolder,eachFileData))
                    subset_score = depth_first_subgraph_detection(graph, att)
                    prec,recall,truescore = printstat(graph,att,true_subgraph,subset_score,eachFileData)
                    timeV = time.time() - start_time
                    file.write(str(prec)+" "+str(recall)+" "+str(timeV)+" "+str(eachFileData.split("_")[2])+" "+str(eachFileData.split("_")[5].split(".")[0]+" "+str(data_source)+"\n"))
                    print("--- %s seconds ---" % (time.time() - start_time))
                    file.close()

      
def main():
    #unittest()
    #test_traffic_data()
    # test_traffic_data()
    #hours = ['01','02']
    #hours = ['03','04']
    #hours = ['05','06']
    hours = ['01']
    #hours = ['01','02','03','04','05','06','07','08']
    test_water_pollution_data_noise_x('0000',hours)
    return 
   
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
        graph[line]=[]
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
    Pvalue=[]
    f=open(os.path.join(froot,'pvalues.dat'),'r')
    s=f.readline()
    while len(s)>0:
        s=s.strip().split(' ')
        Pvalue.append([float(s[slice])])
        s=f.readline()
    return Pvalue
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
    #unittest()
    
    froot = os.path.join(sys.argv[1],'input')
    outroot= os.path.join(sys.argv[1],'output')
    npss='BJ'
    G=genG(froot)
    slices=getSlices(froot)
    result=[]
    
    startTime = time.time()
    print 'start processing : '
    for slice in range(1,slices):
        print 'slice=' +str(slice)+'...'
        Pvalue=addPvalue(froot,slice)
        #print Pvalue
        #G = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
        #Pvalue = [[0.05], [0.16], [0.16], [0.16], [0.16], [0.05]]
        #print Pvalue
        subset_score= depth_first_subgraph_detection(G, Pvalue)
        resultNodes=subset_score[0]
        score=subset_score[1]
        result.append((resultNodes,score))
    runningTime = time.time() - startTime
    print 'finishing ,duration: '+ str(runningTime)
    
    fw=open('DepthFirstScan_'+datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')+'.txt','w+')
    for each in result:
        resultNodes=each[0]
        score=each[1]
        fw.write(str([str(each) for each in resultNodes]))
        fw.write('\n')
        fw.flush()
    fw.close()

