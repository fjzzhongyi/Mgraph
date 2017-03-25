__author__ = 'fengchen'
import numpy as np
import copy
import os
import time
import sys

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
    if compdict.has_key(nid):
        return len(compdict[nid])
    else:
        if att[nid][0] <= alpha:
            return 1
        else:
            return 0


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

#S contains the pvalue and id
def bjScore(subset, att, compdict = {}, alpha = 0.05):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    numOfAbnormalnodes = len([item for item in S if item[1] <= alpha])
    numOfNormalNodes = len(S) - numOfAbnormalnodes
    nn = numOfNormalNodes*1.0
    abn = numOfAbnormalnodes*1.0
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

def npss_score(subset, att, npss, compdict = {}, alpha = 0.05):
    if npss == 'BJ':
        return bjScore(subset, att, compdict = {}, alpha = 0.05)
    else:
        print 'npss score fails.'
        sys.exit()
        return None

# Each node has a single attribute p-value
def BJ_Score(subset, att, compdict = {}, alpha = 0.05):
    n = 0
    nalpha = 0
    for nid in subset:
        if att[nid][0] <= alpha:
            c = getcont(nid, att, compdict, alpha)
            nalpha += c
            n += c
        else:
            n += 1
    return n * KL(nalpha / n, alpha)

def BJ_Score_1(subset, att, alpha = 0.05):
    n = len(subset) * 1.0
    nalpha = len([att[nid][0] for nid in subset if att[nid][0] <= alpha])
    return n * KL(nalpha / n, alpha)


# we only conisder one attribut (p-value)
def nodepriority(nid, att, compdict = {}, alpha = 0.05):
    if att[nid][0] <= alpha:
        return getcont(nid, att, compdict, alpha)
    else:
        return 0
    # return -1 * att[nid][0]

def setprioity(subset, att, compdict = {}, alpha=0.05):
    n = 0
    for i in subset:
        if att[i][0] <= alpha:
            n += getcont(i, att, compdict, alpha)
        else:
            n -= 1
    return n

def refine_graph(graph1, att1, alpha):
    # return graph1, att1, {}
    graph = copy.deepcopy(graph1)
    att = copy.deepcopy(att1)
    bigcomp = []
    comps = []
    for nid, neis in graph.items():
        if att[nid][0] <= alpha and nid not in bigcomp:
            comp = [nid]
            queue = [nid]
            while queue:
                i = queue.pop()
                for j in graph[i]:
                    if att[j][0] <= alpha and j not in comp:
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
                    graph[j].append(n)
            graph[i] = []
            att[i][0] = 1
        graph[n] = list(set(neis))
        att.append([alpha])
        compdict[n] = comp
        n += 1
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
        fs = n * KL(np / n, alpha)
        if not subset_score or subset_score[1] < fs:
            subset_score = [subset, fs, alpha]

        LBound = fs
        UBound = (n + np1) * KL((np + np1) / (n + np1), alpha)
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
    return seeds
    
# graph: {node_id: [node_ids]}. att: {node_id: value}
# Identify seeds nodes that have higher priorities than their neighbors
def depth_first_subgraph_detection(graph, att,npss, radius = 7, anomaly_ratio = 0.5, minutes = 30, alpha_max = 0.15):
    if not radius:
        radius = len(graph)
    start_time = time.time()
    subset_score = None
    alphas = set(item[0] for item in att if item[0] != 0 and item[0] < alpha_max)
    alphas = sorted(list(alphas), key = lambda item: item * -1)
    ori_graph = graph
    ori_att = att
    for alpha_i, alpha in enumerate(alphas):
        graph, att, compdict = refine_graph(ori_graph, ori_att, alpha)
        nodeprio = [0] * len(graph)
        for nid in range(len(graph)):
            nodeprio[nid] = nodepriority(nid, att, compdict, alpha)
        seeds = get_seeds(graph, att, compdict, alpha)
        for seed_i, seed in enumerate(seeds[:5]):
            if not subset_score or seed[0] not in subset_score[0] or seed_i == 0:
                # print 'compdict[seed[0]]', compdict[seed[0]]
                route = Route(seed[0], graph, att, nodeprio, compdict, alpha)
                #print seed
                priqueue = Queue([route, 1])
                if not subset_score:
                    subset_score = [route.getsubset(), BJ_Score(route.getsubset(), att, compdict), alpha]
                iter = 0
                while priqueue.len() > 0:
                    [route, pri] = priqueue.pop()
                    iter += 1
                    neis = route.get_neis()
                    if iter % 5000 == 0:
                        #print 'iterations: ', iter
                        #print 'pop:', route.path, route.incl.keys()
                        duration = (time.time() - start_time)
                        if duration > minutes * 60:
                            #print 'time limit reached... '
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
    return subset_score
 
def read_APDM_data_waterNetwork(path):
    graph = {}
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
    return graph, att, true_subgraph

def prec_recall_fmeasure(detect_subgraph, true_subgraph):
    n = 0.0
#    print detect_subgraph
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
  
def finalTest_DepthFirst_multiCore_waterNetwork(numOfThreads,resultFileName, radius, anomaly_rate, minutes, alpha_max, rootFolder,controlFileList):
    resultGraphFolder = '../../ComparePreRecROC/WaterNetwork_12500/DepthScan/graphResults/'
    files = []
    with open(controlFileList) as f:
        for eachLine in f.readlines():
            if eachLine.rstrip().endswith('.txt'):files.append(eachLine.rstrip())
    for fileName in files:
        graph, att,true_subGraph = read_APDM_data_waterNetwork(os.path.join(rootFolder,fileName))
        start_time = time.time()
        print 'start to process : ',fileName
        subset_score = depth_first_subgraph_detection(graph, att,'BJ',radius,anomaly_rate,minutes,alpha_max)
        pre,rec,fmeasure = prec_recall_fmeasure(subset_score[0],true_subGraph)
        runningTime = time.time() - start_time
        print 'finish processing : ',fileName
        bjScore = subset_score[1]
        
        resultNodes = set(subset_score[0])
        trueSubGraph = set(true_subGraph)
        intersect = resultNodes.intersection(trueSubGraph)
        truePositive = len(intersect)*1.0 ;
        falseNegative = len(trueSubGraph)*1.0 - truePositive ;
        falsePositive = len(resultNodes)*1.0 - len(intersect)*1.0 ;
        trueNegative = len(att) - len(trueSubGraph) - falsePositive ;
        tpr = truePositive / (truePositive+falseNegative) ;
        fpr = trueNegative / (falsePositive + trueNegative) ;    
        f = open(resultFileName,'a')
        
        f.write("{0:.6f}".format(0.0)+"\t"+
                   "{0:.6f}".format(bjScore)+"\t"+
                   "{0:.6f}".format(runningTime)+"\t"+
                   "{0:06d}".format(1000)+"\t"+
                   "{0:06d}".format(len(subset_score[0]))+"\t"+
                   fileName.split('.txt')[0]+"\t"+
                   "{0:.6f}".format(0.0)+"\t"+
                   "{0:.6f}".format(pre)+"\t"+
                   "{0:.6f}".format(rec)+"\t"+
                   "{0:.6f}".format(fmeasure)+"\t"
                   "{0:.6f}".format(tpr)+"\t"+
                   "{0:.6f}".format(fpr)+ "\n")
        f.close()
        try: os.stat(resultGraphFolder)
        except: os.mkdir(resultGraphFolder)
        f = open(os.path.join(resultGraphFolder,fileName),'w')
        if len(subset_score[0]) == 0:
            f.write('null')
            f.close()
        else:
            for node in subset_score[0]:
                id = node
                f.write(str(id)+'\n')
            f.close()
def main():
    numOfThreads = 1
    resultFileName = '../../ComparePreRecROC/WaterNetwork_12500/DepthScan/waterNetworkResult_DepthScan.txt'
    rootFolder = '../../realDataSet/WaterData/source_12500'
    controlFileList = '../../ComparePreRecROC/WaterNetwork_12500/controlFileList.txt'
    resultFileName = './DepthScan_runTime_waterNetwork.txt'
    finalTest_DepthFirst_multiCore_waterNetwork(numOfThreads, resultFileName,10,0.5,6000.0,0.15,rootFolder,controlFileList)
    return

if __name__ == "__main__":
    main()
