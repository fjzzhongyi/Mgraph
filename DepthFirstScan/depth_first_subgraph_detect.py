__author__ = 'fengchen'
import numpy as np
import copy
import os
import time

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
def bjscore(subset, att, compdict = {}, alpha = 0.05):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    numOfAbnormalnodes = len([item for item in S if item[1] <= alpha])
    numOfNormalNodes = len(S) - numOfAbnormalnodes
    nn = numOfNormalNodes*1.0
    abn = numOfAbnormalnodes*1.0
    score = (nn + abn) * KL(abn/(nn + abn), alpha)
    #print 'score value is : ',score,' ; number of nodes : ',len(S)
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

def hcscore(subset, att, compdict = {}, alpha = 0.05):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    alpha = alpha
    N_alpha = len([item for item in S if item[1] <= alpha])
    term = (N_alpha*1.0/len(S)*1.0 - alpha)/ math.sqrt(alpha * (1 - alpha))
    return (math.sqrt(len(S)))*term
    
def tippettScore(subset, att, compdict = {}, alpha_max = 0.05):
    return -min(item[1] for item in S)

def simesScore(subset, att, compdict = {}, alpha_max = 0.05):
    copyedS = copy.deepcopy(S)
    copyedS = sorted(copyedS,key=lambda xx:xx[1])
    minValue = 1.1
    for i in range(len(S)):
        AddI = i + 1
        term = copyedS[i][1]/(AddI+0.0)
        if term < minValue:
            minValue = term
    return len(S)*minValue

def fisherScore(subset, att, compdict = {}, alpha_max = 0.05):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    sum = 0.0
    for item in S:
        sum = sum + np.log(item[1])
    return -(sum/len(S))

def stoufferScore(subset, att, compdict = {}, alpha = 0.05):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    sum = 0.0
    for item in S:
        sum = sum + norm.ppf(1-item[1])
    return -sum/np.sqrt(len(S)*1.0)

def pearsonScore(subset, att, compdict = {}, alpha = 0.05):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    maxValue = max(item[1] for item in S)
    return - maxValue/(  math.pow( alpha, 1/(len(S)*1.0) )   )

def edgingtonScore(subset, att, compdict = {}, alpha = 0.05):
    alpha = max(item[1] for item in S)
    if alpha > alpha_max:
        alpha = alpha_max
    sum = 0.0
    for item in S:
        sum = sum + item[1]
    return -sum/(len(S)*1.0)

def npss_score(subset, att, npss, compdict = {}, alpha = 0.05):
    if npss == 'BJ':
        return bjscore(subset, att, compdict = {}, alpha = 0.05)
    elif npss == 'HC':
        return hcscore(subset, att, compdict = {}, alpha = 0.05)
    elif npss == 'Edgington':
        return edgingtonScore(subset, att, compdict = {}, alpha = 0.05)
    elif npss == 'Pearson':
        return pearsonScore(subset, att, compdict = {}, alpha = 0.05)
    elif npss == 'Stouffer':
        return stoufferScore(subset, att, compdict = {}, alpha = 0.05)
    elif npss == 'Fisher':
        return fisherScore(subset, att, compdict = {}, alpha = 0.05)
    elif npss == 'Simes':
        return simesScore(subset, att, compdict = {}, alpha = 0.05)
    elif npss == 'Tippett':
        return tippettScore(subset, att, compdict = {}, alpha = 0.05)
    else:
        print 'npss score fails.'
        sys.exit()
        return None
    

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
                    subset_score = [route.getsubset(), f_score(route.getsubset(), att, compdict), alpha]
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
    fmeasure = 0.0
    if float(prec) < 1e-6 and float(recall) < 1e-6:
        fmeasure = 0.0
    else:
        fmeasure = 2*( prec*recall / (prec+recall) )
    return prec, recall, fmeasure
  
def printstat(graph, att, true_subgraph, subset_score, alpha = 0.01):
    if set(subset_score[0]) == set(true_subgraph):
        prec,recall,fmeasure = prec_recall(subset_score[0], true_subgraph)
        truescore = calc_true_subgraph_score(att, true_subgraph)
        if truescore == subset_score[1]:
            print 'true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1])
        else:
            print 'true score vs detected score: {0} vs {1}'.format(calc_true_subgraph_score(att, true_subgraph), subset_score[1])
    else:
        prec, recall,fmeasure = prec_recall(subset_score[0], true_subgraph)
    return prec,recall,fmeasure


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

def test_traffic_data():
    folders = ['waterData-noise-0', 'GridData', 'GridData1', 'GridData2', 'GridData3']
    for folder in folders[:1]:
        filenames = os.listdir(folder)
        filenames = [filename for filename in filenames if filename.find('readme') < 0]
        for filename in sorted(filenames, key = lambda item: len(item))[:]:
            print os.path.join(folder, filename)
            start_time = time.time()
            graph, att, true_subgraph = read_APDM_data(os.path.join(folder, filename))
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
        for eachFileData in os.listdir(os.path.join(folderName,eachSubFolder)):
            f = open("additive_result.txt",'a')
            start_time = time.time()
            graph, att,true_subgraph = read_APDM_data(os.path.join(folderName,eachSubFolder,eachFileData))
            subset_score = depth_first_subgraph_detection(graph, att)
            prec,recall = printstat(graph,att,true_subgraph,subset_score)
            print eachFileData
            print prec,recall
            timeV = time.time() - start_time
            f.write(str(eachFileData.split("_")[5].split(".")[0]+" "+str(prec)+" "+str(recall)+" "+str(timeV)))
            print("--- %s seconds ---" % (time.time() - start_time))
            f.close()

def read_APDM_data(path):
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

def read_APDM_data_trans(path):
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


def test_single_case(APDMFileName,npss,outPutFileName,noiseLevel = 'none', hour = 'none', alpha_max = 0.15, verbose_level = 0):
    start_time = time.time()
    if verbose_level == 1:
        print '--------------------------------------------------------------------------------'
        print 'processing file : ', os.path.join(APDMfileName)
    if verbose_level == 1:
        print 'processing score : ', npss
    graph, att,true_subgraph = read_APDM_data(APDMFileName)
    subset_score = depth_first_subgraph_detection(graph, att,npss)
    prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
    if verbose_level <= 1:
        print '--------------------------------------------------------------------------------'
    bjscore = subset_score[1]
    timeV = time.time() - start_time
    file = open("depth_first_result_water_data_minutes_30.txt",'a')
    file.write("{0:.6f}".format(prec)+"\t"+"{0:.6f}".format(recall)+"\t"+"{0:.6f}".format(fmeasure)+"\t"+
               "{0:.6f}".format(bjscore)+"\t"+"{0:.6f}".format(timeV)+"\t"+hour+"\t"+ noiseLevel + "\t"+'source_12500'+"\n")
    file.close()
                    
def test_water_pollution_data_noise_x(x,hours):
    folderName = '../../realDataSet/WaterData'
    subFolderNames = os.listdir(folderName)
    print subFolderNames
    subFolderNames = [fileName for fileName in subFolderNames if os.path.isdir(folderName+'/'+fileName) ]
    for eachSubFolder in subFolderNames:
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
                    graph, att,true_subgraph = read_APDM_data(os.path.join(folderName,eachSubFolder,eachFileData))
                    subset_score = depth_first_subgraph_detection(graph, att,npss)
                    prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
                    
                    bjscore = subset_score[1]
                    timeV = time.time() - start_time
                    hour = str(eachFileData.split("_")[2])
                    noiseLevel = str(eachFileData.split("_")[5].split(".")[0])
                    data_source = str(data_source)
                    file = open("depth_first_result_water_data_minutes_test.txt",'a')
                    file.write("{0:.6f}".format(prec)+"\t"+"{0:.6f}".format(recall)+"\t"+"{0:.6f}".format(fmeasure)+"\t"+
                               "{0:.6f}".format(bjscore)+"\t"+"{0:.6f}".format(timeV)+"\t"+ noiseLevel+"\t"+npss+"\n")
                    file.close()
                    
def test_grid_data(folderName,data):
    for eachFileData in os.listdir(folderName):
        print 'processing file : ', os.path.join(folderName,eachFileData)
        alpha_max = 0.15
        npssArr = ['BJ','HC','Edgington','Pearson','Stouffer','Fisher','Simes','Tippett']
        for npss in npssArr:
                print 'processing score : ', npss   
                start_time = time.time()
                graph, att,true_subgraph = read_APDM_data(os.path.join(folderName,eachFileData))
                subset_score = depth_first_subgraph_detection(graph, att,npss)
                prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
                timeV = str(time.time() - start_time)
                score = str(subset_score[1])
                noiseLevel = str(eachFileData.split("_")[1].split(".")[0])
                file = open("./depth_result/depth_first_result_grid_100_"+data+".txt",'a')
                file.write(str(prec)+" "+str(recall)+" "+str(fmeasure)+" "+score + " "+ str(timeV) + " "+noiseLevel+" "+npss+"\n")
                file.close()
    print 'finish'
    return

def debugSingleCase():
    fileName = './APDM-Water-source-12500_time_02_hour_noise_6.txt'
    noiseLevel = '02'
    hour = '6'
    npss = 'BJ'
    data_source = 'source_12500'
    print 'current processing file : ', fileName
    start_time = time.time()
    graph, att,true_subgraph = read_APDM_data(fileName)
    subset_score = depth_first_subgraph_detection(graph, att,npss)
    prec,recall,fmeasure = printstat(graph,att,true_subgraph,subset_score)
    bjscore = subset_score[1]
    timeV = time.time() - start_time
    print 'precision : ',prec, ' ; recall : ',recall,
    file = open("depth_first_result_water_data_minutes_30.txt",'a')
    file.write("{0:.6f}".format(prec)+"\t"+"{0:.6f}".format(recall)+"\t"+"{0:.6f}".format(fmeasure)+"\t"+
               "{0:.6f}".format(bjscore)+"\t"+"{0:.6f}".format(timeV)+"\t"+ noiseLevel+"\t"+npss+"\n")
    file.close()

def test_water_dataSet():
    rootFolderName = '../../realDataSet/WaterData/source_12500/'
    from multiprocessing import Pool
    pool = Pool(processes=10)
    for eachDataFile in os.listdir(rootFolderName):
        outPutFileName = './kddGreedy_result_source_12500.txt'
        noiseLevel = eachDataFile.split('_')[5].split('.')[0]
        hour = eachDataFile.split('_')[2]
        InputAPDM = os.path.join(rootFolderName,eachDataFile)
        print 'processing file : ',InputAPDM
        pool.apply_async(test_single_case, args=(InputAPDM, 'BJ', outPutFileName,noiseLevel,hour,) )
    pool.close()
    pool.join()

def test_single_case_trans(APDMFileName,npss,outPutFileName, alpha_max = 0.15,verbose_level = 0):
    start_time = time.time()
    if verbose_level == 1:
        print '--------------------------------------------------------------------------------'
        print 'processing file : ', os.path.join(APDMFileName)
    if verbose_level == 1:
        print 'processing score : ', npss
    #graph : adj list dict() ; att : list ; globalPValue : list
    return
    
def test_trans_dataset():
    rootFolderName = '../../realDataSet/TransportationCandidate2'
    from multiprocessing import Pool
    pool = Pool(processes=1)
    for eachDataSet in os.listdir(rootFolderName):
        for eachDataFile in os.listdir(rootFolderName+'/'+eachDataSet):
            outPutFileName = './pdeth_first_2013-07_result.txt'
            InputAPDM = rootFolderName+'/'+eachDataSet+'/'+eachDataFile
            print 'processing file : ',InputAPDM
            pool.apply_async(test_single_case_trans, args=(InputAPDM, 'BJ', outPutFileName) )
    pool.close()
    pool.join()
       
def main():
    #test_water_dataSet()
    #hours = ['01','02','03','04','05','06','07','08']
    #test_water_pollution_data_noise_x('8',hours)
    #hours = ['01','02','03','04','05','06','07','08']
    #test_water_pollution_data_noise_x('2','02')
    #debugSingleCase()
    test_trans_dataset()
    return
    #data = ['data1','data2','data3','data4','data5','data6','data7','data8','data9','data10',]
    #for dataset in data:
    #    test_grid_data('../simuDataSet/'+dataset,dataset)
    #test_grid_data('../simuDataSet/data1')
    #return

if __name__ == "__main__":
    main()
