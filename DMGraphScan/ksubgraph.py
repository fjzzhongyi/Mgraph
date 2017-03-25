# created  May 3

import ctypes
import os
import sys
import math
import copy
import random
import time
import unittest
from UnionFind import UnionFind
from os.path import join
from node import Node
sys.setrecursionlimit(25000)

INF = 1e+10
EPS = 1e-6
#ALPHA = 0.15

#_mod = ctypes.cdll.LoadLibrary('./libmckp.so')
#_mckp_01 = _mod.mckp_01
#_mckp_01.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
#_mckp_01.restype = ctypes.POINTER(ctypes.c_int)

def mckp_01(values, K):

    val = _mckp_01(((ctypes.c_int)*len(values))(*values), K)
    if val[0] == -1:
        return []
    else:
        return [val[i] for i in range(values[0] + 2)]

def bruteforce(data, K):
    
    d = copy.deepcopy(data)
    num = [0 for i in range(len(d))]
    len_num = len(d)
    cat = {}
    while len_num > 0 and num[len_num-1] <= len(d[len_num-1]):
        w = 0
        p = 0
        for i in range(len_num):
            if num[i] < len(d[i]):
                w += d[i][num[i]][2]
                p += d[i][num[i]][1]
        if w <= K:
            if p not in cat:
                cat[p] = []
            cat[p].append([[num[i] for i in range(len_num)], w])
        num[0] += 1
        for i in range(len_num):
            if num[i] > len(d[i]) and i+1 < len_num:
                num[i] = 0
                num[i+1] += 1
    if len(cat) > 0:
        max_p = max(cat.keys())
        ls = cat[max_p]
        temp = sorted(ls, key=lambda xx:xx[1])
        num = temp[0][0]
        return [temp[0][1], [d[i][num[i]][0] for i in range(len_num) if num[i] < len(d[i])]]
    else:
        return []

"""
class Node:
    def __init__(self, name, pvalue, nchild, child):
        self.name = name
        self.pvalue = pvalue
        self.nchild = nchild
        self.child = child
"""
class Optim:
    def __init__(self, alpha, Gpv, app, curi, Fi=0):
        self.alpha = alpha
        self.Gpv = Gpv
        self.app = app
        self.curi = curi
        self.Fi = 0
        self.q = 0
    def setq(self):
        self.q = 0
        for item in self.Gpv:
            if self.Gpv[item] < self.alpha[self.curi]:
                self.q += 1

def BJ(alpha, N_a, N):

    if N == 0:
        return 0
    aa = N_a * 1.0 / N
    bb = alpha
    return N * (aa*math.log((aa/bb) + EPS) + (1-aa)*math.log(((1-aa)/(1-bb)) + EPS))

def HC(alpha, N_a, N):

    return (N_a - N*alpha) / (math.sqrt(N*alpha*(1-alpha)))

def g(v, alpha):

    if v.pvalue < alpha:
        return 0
    else:
        return 1

def maxDeV(v, l, DeS, DeA, DeV):

   temp = []
   if l in DeS[v.name]:
       temp.append(DeS[v.name][l])
   if l in DeA[v.name]:
       temp.append(DeA[v.name][l])
   if len(temp) > 0:
       DeV[v.name][l] = max(temp) 

def maxwsum(childb):

    wsum2 = 0
    dichild = {}
    count = childb[0]
    for i in range(childb[0]):
	dichild[i] = {}
	for j in range(childb[1+i]):
	    temp = []
	    count += 1
	    temp.append(childb[count])
	    count += 1
	    temp.append(childb[count])
	    count += 1
	    temp.append(childb[count])
	    dichild[i][j] = temp

    for j in range(childb[0]):
	last = dichild[j][childb[1+j]-1]
	for i in range(childb[1+j]-1):
            if dichild[j][i][1] >= last[1]:
                if dichild[j][i][1] > last[1] or dichild[j][i][2] < last[2]:
		    last = dichild[j][i]
        wsum2 += last[2]
    return wsum2
	    

def maxtree(v, K, DeS, DeA, DeV, isbrute, galpha, opt, x):

    terval = 0
    for i in range(v.nchild):
        temp = maxtree(v.child[i], K, DeS, DeA, DeV, isbrute, galpha, opt, x)
        if temp[0] == -1:
            return temp
        terval += temp[0]

    print '[maxtree] current node:', v.name

    if v.name not in DeS:
        DeS[v.name] = {}
    if v.name not in DeA:
        DeA[v.name] = {}
    if v.name not in DeV:
        DeV[v.name] = {}
    
    gv = g(v, galpha)
    
    if v.nchild == 0:
        DeA[v.name][gv] = 1 - gv
        DeV[v.name][gv] = 1 - gv
    else:
        childa = []
        childmckp = []
        map = {}
        childcount = 0
        childdt = {i:0 for i in range(v.nchild)}
        for i in range(v.nchild):
            temp = []
            isfirst = True
            for j in range(K+1):
                if j in DeA[v.child[i].name]:
                    temp.append([str(i)+'_'+str(j), DeA[v.child[i].name][j], j])

                    if isfirst:
                        isfirst = False
                        childmckp.extend([childcount, 0, 0])
                        map[childcount] = 'dummy'
                        childcount += 1
                        childdt[i] += 1

                    childmckp.extend([childcount, DeA[v.child[i].name][j], j])
                    map[childcount] = str(i)+'_'+str(j)
                    childcount += 1
                    childdt[i] += 1
            if len(temp) > 0:
                childa.append(temp)

        if not isbrute:
            childb = [len(childa)]
            for i in range(v.nchild):
                if childdt[i] == 0:
                    continue
                childb.append(childdt[i])
            childb.extend(childmckp)
            wsum2 = maxwsum(childb)

        for l in range(K+1):

            temp = [DeV[vi][l] for vi in [v.child[i].name for i in range(v.nchild)] if l in DeV[vi]]
            if len(temp) > 0:
                DeS[v.name][l] = max(temp)

            if isbrute:
                ret = bruteforce(childa, l-gv)
            else:
		temp = []
		#print '[maxtree] wsum2:', wsum2, ' K = ', K, ' l-gv = ', l-gv, childb
		if wsum2 >= l-gv and l-gv >= 0:
                    temp = mckp_01(childb, l-gv)
		#print '[maxtree] mckp:', temp
                classes = len(childa)
                ret = []
                if len(temp) > 0:
                    ret = [temp[classes+1], [map[temp[i]] for i in range(classes) if map[temp[i]] != 'dummy']]
		    #print '[maxtree] mckp ret:', ret
            
            if len(ret) > 0:
                lmax = gv + ret[0]
                lval = 1 - gv
                lx = {}
                for idx in range(len(ret[1])):
                    i = int(ret[1][idx].split('_')[0])
                    j = int(ret[1][idx].split('_')[1])
                    lval += DeA[v.child[i].name][j]
                    lx[v.child[i]] = j
                if lmax not in DeA[v.name] or lval > DeA[v.name][lmax]:
                    DeA[v.name][lmax] = lval
                    if v not in x:
                        x[v] = {}
                    if lmax not in x[v]:
                        x[v][lmax] = lx
                    maxDeV(v, lmax, DeS, DeA, DeV)
		#non optimize
		#DeA[v.name][l] = lval
		#x[v][l] = lx
            maxDeV(v, l, DeS, DeA, DeV)

    temp = [DeA[v.name][l] for l in DeA[v.name]]
    terval += 1 - g(v, galpha)

    if False and opt.curi > 0 and len(temp) > 0:
        opt.q = opt.q - (terval - max(temp))
        phiscore = 0
        if opt.q > 0:
            if opt.app == 'bj':
                phiscore = BJ(opt.alpha[opt.curi], opt.q, opt.q)
            if opt.app == 'hc':
                phiscore = HC(opt.alpha[opt.curi], opt.q, opt.q)
        if phiscore < opt.Fi:
            return [-1, v]

    return [terval, v]


def getmaxk(v, K, DeV):

    so = [-1, 0]
    for i in range(K+1):
        if i not in DeV[v.name]:
            continue
        if DeV[v.name][i] > so[0]:
            so[0] = DeV[v.name][i]
            so[1] = i
    return so[1]
        

def getsubtree(v, K, x):
    
    S = []
    print '[getsubtree] current node:', v.name, v.pvalue, K

    if v in x and K in x[v]:
        for vchild in x[v][K].keys():
            S.extend(getsubtree(vchild, x[v][K][vchild], x))
            
    if v.name not in S:
        S.append(v.name)
    return S
    

def KCardTree(vlis, K, DeS, DeA, DeV, isbrute, galpha, opt, x, ismax=True):

    if ismax:
        steiner = maxtree(vlis[0], K, DeS, DeA, DeV, isbrute, galpha, opt, x)
        vlis[0] = steiner[1]
    v = vlis[0]

    k = getmaxk(v, K, DeV)
    while k in DeS[v.name] and k in DeA[v.name] and DeS[v.name][k] > DeA[v.name][k] or k not in DeA[v.name]:
        temp = []
        for vi in v.child:
            if k in DeV[vi.name]:
                temp.append([DeV[vi.name][k], vi])
        so = [-1, []]
        for i in range(len(temp)):
            if temp[i][0] > so[0]:
                so[0] = temp[i][0]
                so[1] = temp[i][1]
        if so[0] == -1:
	    break
        v = so[1]

    return getsubtree(v, k, x)


def MinimumSpanningTree(G):
    for u in G:
        for v in G[u]:
            if G[u][v] != G[v][u]:
                raise ValueError("MinimumSpanningTree: asymmetric weights")

    subtrees = UnionFind()
    tree = []
    for W,u,v in sorted((G[u][v],u,v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u,v))
            subtrees.union(u,v)
    return tree

def SubTree(v, edges, Gpv, colornode):
    i = len(edges) - 1
    nchild = 0
    childlis = []

    while i >= 0:
        if i in colornode:
            i -= 1
            continue
        if v == edges[i][0]:
            child = edges[i][1]
            colornode[i] = 1
            nchild += 1
            childlis.append(SubTree(child, edges, Gpv, colornode))
        if v == edges[i][1]:
            child = edges[i][0]
            colornode[i] = 1
            nchild += 1
            childlis.append(SubTree(child, edges, Gpv, colornode))
        i -= 1

    treenode = Node(str(v), Gpv[v], nchild, childlis)
    return treenode


"""
MAIN procedure
Input G, Gpv and Ground Truth
"""

#############Input G, Gpv and GroundTruth###################
def generateG(fn, folder):

    G = {}
    data = open(join('./data/', 'edgeset.dat'), 'r').read().split('\n')
    for item in data:
	if len(item) == 0:
	    continue
        edge = item.split(' ')
        if int(edge[0]) not in G:
            G[int(edge[0])] = {}
        G[int(edge[0])][int(edge[1])] = 1
        if int(edge[1]) not in G:
            G[int(edge[1])] = {}
        G[int(edge[1])][int(edge[0])] = 1
    return G

"""
def generateG():

    G = {}
    #data = open('/home/zhoubj/APDM/APDM/realDataSet/WaterData/Graph-WaterRealGraph.txt', 'r').read().split('\r\n')
    data = open('/home/zhoubj/BackUp/APDM/APDM/realDataSet/WaterData/Graph-WaterRealGraph.txt', 'r').read().split('\r\n')
    for item in data:
        if len(item) == 0:
            continue
        edge = item.split(' ')
        if int(edge[0]) not in G:
            G[int(edge[0])] = {}
        G[int(edge[0])][int(edge[1])] = 2
        if int(edge[1]) not in G:
            G[int(edge[1])] = {}
        G[int(edge[1])][int(edge[0])] = 2
    return G
"""

def generateGpv(fn, folder):

    Gpv = {}
    fstr = join(folder, fn)
    data = open(fstr, 'r').read().split('\n')
    for item in data:
	if len(item) == 0:
	    continue
        node = item.split(' ')
        Gpv[int(node[0])] = float(node[1])

    return Gpv

def generateGt(fn, folder):

    Gt = {}
    fstr = join(folder, fn)
    data = open(fstr, 'r').read().split('\n')
    isdata = False
    last = ''
    for item in data:
	if 'SECTION4' in last and item == 'EndPoint0 EndPoint1 Weight':
	    isdata = True
	    continue
	last = item
	if not isdata:
	    continue
        if 'END' in item:
	    break
        edge = item.split(' ')
	Gt[int(edge[0])] = 1
	Gt[int(edge[1])] = 1
    return Gt

############################################################

def getbfstree(G, Gpv, v, T):

    nchild = 0
    childlis = []

    temp = []
    for vi in G[v]:
        if vi not in T:
            T[vi] = 1
            temp.append(vi)

    #print g.nodes()
    for vi in temp:
        nchild += 1
        item = getbfstree(G, Gpv, vi, T)
        childlis.append(item)
    treenode = Node(str(v), Gpv[v], nchild, childlis)
    return treenode

def IBFS(G, Gpv, v):

    T = {v:1}
    return getbfstree(G, Gpv, v, T)

def IRST(G, Gpv, v):

    for i in G:
        for j in G[i]:
            temp = random.random()
            G[i][j] = temp
            G[j][i] = temp
    treeedges = MinimumSpanningTree(G)
    treenode = SubTree(v, treeedges, Gpv, {})
    return treenode

def dijstg(G, src, previous, d):
    if src not in G:
        raise TypeError('the root of the shortest path tree cannot be found in the G')
    for v in G:
        d[v] = float('inf')
    d[src] = 0
    Q = {v:1 for v in G}
    while len(Q) > 0:
        temp = {v:d[v] for v in Q}
        u = min(temp, key=temp.get)
        del Q[u]
        for v in G[u]:
            if v not in Q:
                continue
            new_distance = d[u] + G[u][v]
            if new_distance < d[v]:
                d[v] = new_distance
                previous[v] = u

def ISTG(G, Gpv, vroot, galpha):
#
    print 'ISTG...'
    v = vroot
    #G1 is the copy of G, means that G1 will be modified
    G1 = copy.deepcopy(G)
    #S is set of nodes that is anomalous
    S = {v:1 for v in Gpv if Gpv[v] < galpha}

    s0 = -1
    for u in G1:
        if u in S:
            G1[u][s0] = 0
        for v in G1[u]:
            G1[u][v] = 1
    G1[s0] = {}
    for v in S:
        G1[s0][v] = 0

    #print S
    #print galpha

    previous = {}
    d = {}
    dijstg(G1, s0, previous, d)
    N_S = {v:{v:[0, [v]]} for v in S}
    com = [v for v in G if v not in S]
    for v in com:
        dis = d[v]
        path = [v]
        if v in previous:
            u = previous[v]
        else:
            continue
        while u not in S:
            path.append(u)
            u = previous[u]
        path.append(u)
        N_S[u][v] = [dis, path]

    maps = {}
    for u in N_S:
        for v in N_S[u]:
            maps[v] = u

    G2 = {}
    for u in G1:
        for v in G1[u]:
            if u == s0 or v == s0 or u == v or u not in maps or v not in maps:
                continue
            if maps[u] == maps[v]:
                continue
            if maps[u] not in G2:
                G2[maps[u]] = {}
            if maps[v] not in G2[maps[u]]:
                G2[maps[u]][maps[v]] = []
            temp = [item for item in N_S[maps[u]][u][1]]
            temp1 = [item for item in N_S[maps[v]][v][1]]
            temp.reverse()
            #temp1.reverse()
            temp.extend(temp1)
            if temp is None:
                del G2[maps[u]][maps[v]]
                continue
            G2[maps[u]][maps[v]].append([N_S[maps[u]][u][0] + G[u][v] + N_S[maps[v]][v][0], temp])
    G2dis = copy.deepcopy(G2)
    for u in G2:
        for v in G2[u]:
            if len(G2[u][v]) > 0:
                idx = float('inf')
                for item in G2[u][v]:
                    if item[0] < idx:
                        G2dis[u][v] = item[1]
                        idx = item[0]
                G2[u][v] = idx

    treeedges = MinimumSpanningTree(G2)

    G3 = {}
    for edge in treeedges:
        temp = []
        print edge, G2dis[edge[0]][edge[1]]
        if edge[1] in G2dis[edge[0]] and len(G2dis[edge[0]][edge[1]]) > 0:
            temp.extend(G2dis[edge[0]][edge[1]])
        for i in range(len(temp)-1):
            item = temp[i]
            if item not in G3:
                G3[item] = {}
            G3[item][temp[i+1]] = G1[item][temp[i+1]]
            if temp[i+1] not in G3:
                G3[temp[i+1]] = {}
            G3[temp[i+1]][item] = G1[temp[i+1]][item]

    treeedges = MinimumSpanningTree(G3)

    treenode = SubTree(vroot, treeedges, Gpv, {})
    return treenode


def isna(pv, galpha):

    if pv < galpha:
        return 1
    else:
        return 0

def dij(G, Gpv, src, app, previous, d, galpha):
    if src not in G:
        raise TypeError('the root of the shortest path tree cannot be found in the G')
    for v in Gpv:
        d[v] = [float('inf'), min(Gpv[v], galpha), isna(Gpv[v], galpha), 1]
    d[src][0] = 0
    Q = {v:1 for v in Gpv}
    while len(Q) > 0:
        temp = {v:d[v][0] for v in Q}
        u = min(temp, key=temp.get)
        del Q[u]
        for v in G[u]:
            if v not in Q:
                continue
            alpha = max(d[u][1], min(Gpv[v], galpha))
            N_a = d[u][2] + isna(Gpv[v], galpha)
            N = d[u][3] + 1
            if app == 'bj':
                new_distance = [math.exp(-BJ(alpha, N_a, N)), alpha, N_a, N]
            if app == 'hc':
                new_distance = [math.exp(-HC(alpha, N_a, N)), alpha, N_a, N]
            if new_distance[0] < d[v][0]:
                d[v] = new_distance
                previous[v] = u

def getgeotree(spantree, Gpv, v):

    nchild = 0
    childlis = []
    for u in spantree[v]:
        childlis.append(getgeotree(spantree, Gpv, u))
        nchild += 1
    treenode = Node(v, Gpv[int(v)], nchild, childlis)
    return treenode

def IGEO(G, Gpv, rootg, app, galpha):

    spantree = {rootg:{}}

    distances = {}
    predecessors = {}
    dij(G, Gpv, rootg, app, predecessors, distances, galpha)
    for v in predecessors:
        u = predecessors[v]
        if u not in spantree:
            spantree[u] = {}
        if v not in spantree:
            spantree[v] = {}
        spantree[u][v] = 1

    treenode = getgeotree(spantree, Gpv, rootg)
    return treenode

#galpha = 0
#x = {}

def budgetK(fn, folder, apptree, isOPTK):
    foutn = fn
    if isOPTK:
        foutn += '_' + apptree + '_opt.dat'
	foutn = apptree.upper() + '_OPT/' + foutn
    else:
        foutn += '_' + apptree + '_noopt.dat'
	foutn = apptree.upper() + '_NOOPT/' + foutn
    f = open(foutn, 'w')

    myfile = ''
    CC = 2
    UALPHA = [0.15]
    KCONSTRAINT = 31
    ALPHAMAX = 0.15
    app = 'bj'
    isbrute = False
    #folder = '/home/zhoubj/BackUp/APDM/APDM/realDataSet/WaterData/'
    #folder = '/home/zhoubj/BackUp/APDM/APDM/realDataSet/WaterData/source_12500/'
    #folder = '/home/zhoubj/BackUp/APDM/APDM/realDataSet/WaterData/source_5421/'
    #folder = '/home/zhoubj/BackUp/APDM/APDM/realDataSet/WaterData/source_5420/'
    #G = generateG()
    G = generateG(fn, folder)
    Gpv = generateGpv(fn, folder)
    #Gt = generateGt(fn, folder)

    """
    print fn
    print '#edges:', len(G)
    print '#Gpv:', len(Gpv)
    print '#true subgraph:', len(Gt)
    print Gt.keys()
    pvlis = {Gpv[v]:0 for v in Gpv if Gpv[v] < ALPHAMAX}
    pvlis = pvlis.keys()
    pvlis.sort()
    print pvlis
    del pvlis[0]
    pvlis.append(ALPHAMAX)
    print pvlis
    UALPHA = pvlis
    """



    myfile += '\n\n' + fn + '\n'
    myfile += 'Graph: ' + str(len(G)) + ' nodes\n'
    myfile += 'Ground Truth: subgraph has ' + str(1) + ' nodes\n\n'
    myfile += 'The precision recall value for approximate algorithm\n'
    NPSS = []
    RUNTIME = []

    for cc in range(CC):
        opt = Optim(UALPHA, Gpv, app, 0)
        for ialpha in range(len(UALPHA)):
            galpha = UALPHA[ialpha]
            opt.curi = ialpha
            opt.setq()

            temp = [v for v in Gpv if Gpv[v] < galpha]
            if len(temp) == 0:
                continue
            rootg = temp[random.randint(0, len(temp)-1)]
	    treebuildstart = time.time()
	    if apptree == 'bfs':
                rootnode = IBFS(G, Gpv, rootg)
	    elif apptree == 'rst':
                rootnode = IRST(G, Gpv, rootg)
	    elif apptree == 'stg':
                rootnode = ISTG(G, Gpv, rootg, galpha)
	    elif apptree == 'geo':
                rootnode = IGEO(G, Gpv, rootg, 'bj', galpha)
	    else:
		print 'app ERROR...'
		sys.exit()
	    treebuildend = time.time()
	    RUNTIME.append([treebuildend-treebuildstart])

            print '[main] Select the root node:', rootg, rootnode.name

            x = {}
            DeS = {}
            DeA = {}
            DeV = {}

            pr = []
            runstart = []
            runend = []
            result = []
            #isOPTK = True
            itr = 0
            KLIS = [i for i in range(KCONSTRAINT)]
            KLIS.reverse()
            rootlis = [rootnode]
            for Kcon in KLIS:
                if not isOPTK:
                    x = {}
                    DeS = {}
                    DeA = {}
                    DeV = {}
                    prundt = {}

                runstart.append(time.time())
                #result.append(KCardTree(rootnode, Kcon, DeS, DeA, DeV, isbrute))

                if Kcon == KCONSTRAINT - 1:
                    result.append(KCardTree(rootlis, Kcon, DeS, DeA, DeV, isbrute, galpha, opt, x))
                else:
                    result.append(KCardTree(rootlis, Kcon, DeS, DeA, DeV, isbrute, galpha, opt, x, False))

                runend.append(time.time() - runstart[len(runstart)-1])

                print 'MCKP: K =', Kcon, ' RESULT:', result[itr]
                print 'RUNNING TIME:', runend[len(runend)-1]
                common = 0
		N_a = 0
                for idx in result[itr]:
                    #if int(idx) in Gt:
                    #    common += 1
		    if Gpv[int(idx)] < galpha:
			N_a += 1

                if len(result[itr]) != 0:
                    pr.append((common*1.0/(len(result[itr])), common*1.0/1))
                else:
                    pr.append((common*1.0/(len(result[itr]+1)), common*1.0/1))

                NPSS.append([[result[itr], pr[itr][0], pr[itr][1]], \
		    BJ(galpha, N_a, len(result[itr])), HC(galpha, N_a, len(result[itr]))])

                i = itr
                print 'K = ', i, 'precision:', pr[i][0], 'recall:', pr[i][1], 'running time:', runend[i]
                myfile += 'K = ' + str(Kcon) + ' precision:' + str(pr[i][0]) + \
                    ' recall:' + str(pr[i][1]) + ' running time:' + str(runend[i]) + '\n'
                itr += 1

            if app == 'bj':
                opt.Fi = max(opt.Fi, NPSS[len(NPSS)-1][1])
            if app == 'hc':
                opt.Fi = max(opt.Fi, NPSS[len(NPSS)-1][2])
	    RUNTIME[cc].append(runend)

    sobj = [-1, []]
    sohc = [-1, []]
    for i in range(len(NPSS)):
        if NPSS[i][1] > sobj[0]:
	    sobj[0] = NPSS[i][1]
	    sobj[1] = NPSS[i][0]
        if NPSS[i][2] > sohc[0]:
	    sohc[0] = NPSS[i][2]
	    sohc[1] = NPSS[i][0]
    if sobj[0] != -1:
        myfile += 'BJ statistic precision:' + str(sobj[1][1]) + ' recall:' + str(sobj[1][2]) + '\n'
    if sohc[0] != -1:
        myfile += 'HC statistic precision:' + str(sohc[1][1]) + ' recall:' + str(sohc[1][2]) + '\n'
    for cc in range(CC):
        if len(RUNTIME) > cc and len(RUNTIME[cc]) > 1:
            myfile += 'Time ' + str(cc) + ' buldtreetime ' + str(RUNTIME[cc][0]) + \
		' getsolutiontime ' + str(sum(RUNTIME[cc][1])) + ' ' + str(RUNTIME[cc][1]) + '\n'

    bjg = sorted(NPSS, key=lambda xx:xx[1])
    bjg.reverse()
    hcg = sorted(NPSS, key=lambda xx:xx[2])
    hcg.reverse()

    myfile += 'graphbj ' + str(bjg[0][0][0]) + '\n'
    myfile += 'scorebj ' + str(bjg[0][1]) + '\n'
    myfile += 'graphhc ' + str(hcg[0][0][0]) + '\n'
    myfile += 'scorehc ' + str(hcg[0][2]) + '\n'

    f.write(myfile)
    f.close()
    return [myfile, folder]

#ISTG(G, Gpv, vroot, galpha)
#node=ISTG({101:{},102:{},103:{}},{101:0.05,102:0.16,103:0.27},101,0.15)   
#print node.child, node.name, node.pvalue