import GV,math,itertools
import solutionlist,gc
import sys,time
from pyspark import SparkContext
sc=None
def sc_start():
    global sc
    sc=SparkContext.getOrCreate()

def extend(ori, root, target):
    if len(ori)==0 and root is None:
        return set()
    on=root.parent
    result=set()|ori
    while on is not None and on !=target:
        result.add(on)
        on=on.parent
    return result


def nll_enc(nll):
    re=[]
    for nodeset in nll:
        re.append(set([node.name for node in nodeset]))
    return re
def nll_dec(nll):
    re=[]
    for nameset in nll:
        re.append(set([GV.name_node[nodename] for nodename in nameset]))
    return re
def rll_enc(rll):
    re=[]
    for node in rll:
        if node is None:
            re.append(None)
        else:
            re.append(node.name)
    return re
def rll_dec(rll):
    re=[]
    for nodename in rll:
        if nodename is None:
            re.append(None)
        else:
            re.append(GV.name_node[nodename])
    return re

def psi (input):
    # input: set of nodes in time GV.gv.slices
    def pv2index(li):
        rl=[]
        for i in range(len(li)):
            if li[i]<=GV.alpha:
                rl.append(i)
        return rl
    def BJ(alpha, N_a, N):
        if N == 0:
            return 0
        aa = N_a * 1.0 / N
        bb = alpha
        EPS=10**-6
        return N * (aa*math.log((aa/bb)+EPS) + (1-aa)*math.log(((1-aa)/(1-bb))+EPS))
    
    # N_a:number of common anomalous nodes    N: total number of nodes 
    N_a=0
    N=0

    for t in range(GV.gv.slices):
        # input[t]: set of nodes at the same time t
        S=set()
        anom_nodes=filter(lambda x: True if x.pvalue[t]<=GV.alpha else False ,input[t])
        N_a+=len(anom_nodes)     
        N+=len(input[t])   
    
    return BJ(GV.alpha,N_a,N)
            

# calc evert vertex's omega
def leaf(node):
    node.Nroot=[[None for i in range(GV.gv.slices)] for j in range(GV.gv.B+1)]
    def psi4nonleaf (input):
        # input: list of 0/1   e.g. (0,1,0,0,1,1)
        def BJ(alpha, N_a, N):
            if N == 0:
                return 0
            aa = N_a * 1.0 / N
            bb = alpha
            EPS=10**-6
            return N * (aa*math.log((aa/bb)+EPS) + (1-aa)*math.log(((1-aa)/(1-bb))+EPS))
        # N_a:number of common anomalous nodes    N: total number of nodes 
        N=N_a=sum(input) 
        return BJ(alpha_b.value,N_a,N)
    def combine (valuelist, alp):
        # this function can produce possible combinations of (E1, E2, E3, ^^ , En), using (0/1, 0/1, 0/1, ^^, 0/1)
        # another way to produce these is to utilize built-in modules itertools
        # itertools.product(range(2),repeat), but here we only mark pvalue<alpha as 1
        # 
        re=[]
        def recur(level,lis):
            if level>len(valuelist):
                re.append(lis)
            else:
                # here choose whether to add by comparing it with alpha
                if valuelist[level-1]<=alp:
                    recur(level+1,lis+[0])
                    recur(level+1,lis+[1])
                else:
                    recur(level+1,lis+[0])
        recur(1,[])
        return re
    
    Nlist=[0 for n in range(0,GV.gv.B+1)]
    OmegaList=[[set() for i in range(GV.gv.slices)] for j in range(GV.gv.B+1)]
  
    # note: filter out lists whose delta> B 
    combinelists=combine(node.pvalue,GV.alpha)
    def calc(each):
        #delta is the number of change times
        delta=0
        #each (0,1,1,0,0)  length=GV.gv.slices
        # 1. calc the change timeskL
        for i in range(0,len(each)-1):
            delta+=abs(each[i]-each[i+1])
        return (delta,each)
    
    # SPARK version
    t1=time.time()
    global sc
    alpha_b=sc.broadcast(GV.alpha)
    B_b=sc.broadcast(GV.gv.B)
    
    new_lists=sc.parallelize(combinelists)\
            .map(calc)\
            .filter(lambda x: True if x[0]<=B_b.value else False)\
            .map(lambda x: (x[0],(x[1],psi4nonleaf(x[1]))))\
            .reduceByKey(lambda a,b: a if a[1]>b[1] else b)\
            .collect()
    t2=time.time()
    print "leaf  Time: %f" %(t2-t1)
    for each in new_lists:
        # each Format: (delta, (list, psi))
        def transform(lis):
            #~~~new version#
            nl=[set() for i in range(GV.gv.slices)]
            for n in range(len(lis)):
                if lis[n]==1:
                    nl[n].add(node)
            return nl
        # setnodes format: [set() for i in range(GV.gv.slices)] 
        Nlist[each[0]]=each[1][1]
        OmegaList[each[0]]=transform(each[1][0])
    
    #update 
    markN=Nlist[0]
    markList=OmegaList[0]
    for j in range(1,GV.gv.B+1):
        if Nlist[j]>=markN:
            markN=Nlist[j]
            markList=OmegaList[j]
        else:
            Nlist[j]=markN
            OmegaList[j]=markList
        
    node.omega=OmegaList
    node.BJ=Nlist
    for b in range(GV.gv.B+1):
        for t in range(GV.gv.slices):
            if node in node.omega[b][t]:
                node.Nroot[b][t]=node
    #node.showOmega()


def nonleaf(node):
    node.Nroot=[[None for i in range(GV.gv.slices)] for j in range(GV.gv.B+1)]
    leaf(node)
    nodes=node.child+[node]
    pieces=len(node.child)+1
    nl=[[set() for i in range(GV.gv.slices)] for j in range(GV.gv.B+1)]
    # Nlist used for recording psi before
    Nlist=[0 for i in range(GV.gv.B+1)]
    RootList=[[None for i in range(GV.gv.slices)] for j in range(GV.gv.B+1)]
   
    """
    li = []
    for number in range(0,GV.gv.B+1):
        #li=[[0,1,0],[1,0,0],[0,0,1]]
        li+=solutionlist.genlist(pieces,number)
    """
    
    def genlist (num): 
        pie=len(nodes)
        res=[]
        level,number,lis,pieces,ic = 1, num, [], pie, num
        stack=[]
        while True:
            if level>pieces:
                if  number==0:   
                    res.append(lis)
                if len(stack)==0:
                    break
                level,number,lis,pieces,ic= stack.pop()
                continue
            if ic>=0:
                
                stack.append([level, number, lis+[], pieces, ic-1])
                level+=1
                number-=ic
                lis+=[ic]
                ic=number
            else:
                if len(stack)==0:
                    break
                level,number,lis,pieces,ic= stack.pop()
        return res
    def spark_search(e):
        # e: [0,3,4,1,0,0,...]
        #method1: try all connected 
        tl=[set() for i in range(gv_b.value.slices)]
        for i in range(gv_b.value.slices):
            for j in range(len(nodes_b.value)):
                tl[i]|=extend(nodes_b.value[j].omega[e[j]][i],nodes_b.value[j].Nroot[e[j]][i],node_b.value)
            tl[i].add(node_b.value)
        def judgeConnect(l):
            #each is a set of node objects
            for each in l:
                s=each.copy()
                #s is the copied set,  if empty, return True
                #s1 is set of nodes that is being visited, 
                #s2 is the set of nodes that *may* be visited next turn
                if len(s)<=1:
                    return True
                #initial s1
                s1=set([s.pop()])
                while len(s)>0:
                    s2=set()
                    for i in s1:
                        s2.add(i.parent)
                        s2|=set(i.nchild)
                    
                    #here prepared for next turn
                    s1= s&s2
                    s=s-s2
                    if len(s1)==0:
                        return False
            return True
        
        cN= psi(tl)
        # default
        countN=cN
        nll=tl
        rll=[node_b.value for i in range(gv_b.value.slices)]
        
        #method2: reserve max BJ of child tree, first of all, find markNode
        for j in range(len(nodes_b.value)):
            cN=nodes_b.value[j].BJ[e[j]]
            if countN< cN:
                countN=cN
                nll=nodes_b.value[j].omega[e[j]]
                rll=nodes_b.value[j].Nroot[e[j]]
        
        # nll, rll should be
        return (sum(e),(countN,nll_enc(nll),rll_enc(rll)))

    # SPARK version
    # maybe genlist could be calc by spark
    
    t1=time.time()
    global sc
    node_b=sc.broadcast(node)
    nodes_b=sc.broadcast(nodes)
    gv_b=sc.broadcast(GV.gv) 
    # spark_result format []  each element: (B,(psi,nll,rll))
    """
    spark_result = sorted(\
        sc.parallelize(range(0,gv_b.value.B+1))\
        .flatMap(genlist)\
        .map(spark_search)\
        .reduceByKey(lambda a,b:a if a[0]>b[0] else b)\
        .collect(),key=lambda x:x[0])
    """
    result = \
        sc.parallelize(range(0,gv_b.value.B+1))\
        .flatMap(genlist)\
        .map(spark_search)\
        .collect()
    t2=time.time()
    print "length of nodes %d ;  time: %f" %(len(nodes),t2-t1)

    spark_result={}
    for each in result:
        if (each[0] not in spark_result) or (each[0] in spark_result and each[1][0]> spark_result[each[0]][0]):
            spark_result[each[0]]=each[1]
    """
    #e:[0,1,0,0,3] each in li
    sc.parallelize(li)\
        .map(spark_search)\
        .reduceByKey(lambda a,b:a if a[0]>b[0] else b)\
        .sortByKey()
    """
    # Spark version ends 
    for number in range(0,GV.gv.B+1):
        #li=[[0,1,0],[1,0,0],[0,0,1]]
        li=solutionlist.genlist(pieces,number)
        #connectness
        #~~
        if number==0:
            countN=0
            nll=[set() for i in range(GV.gv.slices)]
            rll=[None for i in range(GV.gv.slices)]
        else:
            countN=max(Nlist[:number])
            nll=nl[Nlist[:number].index(max(Nlist[:number]))]
            rll=RootList[Nlist[:number].index(max(Nlist[:number]))]
        
        cN=spark_result[number][0] 
        if cN>countN:
            countN=cN
            nll=nll_dec(spark_result[number][1])
            rll=rll_dec(spark_result[number][2])
        
        nl[number]=nll
        Nlist[number]=countN
        RootList[number]=rll
    node.omega=nl
    node.BJ=Nlist
    node.Nroot=RootList
    return
