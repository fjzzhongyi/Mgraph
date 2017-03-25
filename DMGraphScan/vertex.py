import GV,math,itertools
import solutionlist,gc

def extend(ori, root, target):
    if len(ori)==0 and root is None:
        return set()
    on=root.parent
    result=set()|ori
    while on is not None and on !=target:
        result.add(on)
        on=on.parent
    return result


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
    def combine (valuelist, alp):
        # this function can produce possible combinations of (E1, E2, E3, ^^ , En), using (0/1, 0/1, 0/1, ^^, 0/1)
        # another way to produce these is to utilize built-in modules itertools
        # itertools.product(range(2),repeat)
        re=[]
        def recur(level,lis):
            if level>len(valuelist):
                re.append(lis)
            else:
                if valuelist[level-1]<=alp:
                    recur(level+1,lis+[0])
                    recur(level+1,lis+[1])
                else:
                    recur(level+1,lis+[0])
        recur(1,[])
        return re
    
    Nlist=[0 for n in range(0,GV.gv.B+1)]
    OmegaList=[[set() for i in range(GV.gv.slices)] for j in range(GV.gv.B+1)]
  
    for each in combine(node.pvalue,GV.alpha):
        #delta is the number of change times
        delta=0
        #each (0,1,1,0,0)  length=GV.gv.slices
        # 1. calc the change timeskL
        for i in range(0,len(each)-1):
            delta+=abs(each[i]-each[i+1])
        if delta>GV.gv.B:
            continue
        # now use calculated delta to update Nlist
        # each element in Nlist denotes the number of anomalous attributes
        # sum(each) depicts the number of nodes, but now we focus on attributes
        # 
        def transform(lis):
            #~~~new version#
            nl=[set() for i in range(GV.gv.slices)]
            for n in range(len(lis)):
                if lis[n]==1:
                    nl[n].add(node)
            return nl
        # setnodes format: [set() for i in range(GV.gv.slices)] 
        setnodes = transform(each)
        countN=psi(setnodes)
        if Nlist[delta]<countN:
            Nlist[delta]=countN
            OmegaList[delta]=setnodes
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
        #e:[0,1,0,0,3]
        for e in li:
            #i=0,1,2,^^^,nodes-1
            #cN=sum([ nodes[i].omegaN[e[i]]  for i in range(len(e))])
           
            #method1: try all connected 
            tl=[set() for i in range(GV.gv.slices)]
            for i in range(GV.gv.slices):
                for j in range(len(nodes)):
                    tl[i]|=extend(nodes[j].omega[e[j]][i],nodes[j].Nroot[e[j]][i],node)
                tl[i].add(node)
            def judgeConnect(l):
                #each is a set
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
            
            #cn is the number of anomalous attributes
            cN= psi(tl)
            if countN< cN:
                nll=tl
                countN=cN
                rll=[node for i in range(GV.gv.slices)]
            
            #method2: reserve max BJ of child tree, first of all, find markNode
            for j in range(len(nodes)):
                cN=nodes[j].BJ[e[j]]
                if countN< cN:
                    countN=cN
                    nll=nodes[j].omega[e[j]]
                    rll=nodes[j].Nroot[e[j]]
            
        nl[number]=nll
        Nlist[number]=countN
        RootList[number]=rll
    node.omega=nl
    node.BJ=Nlist
    node.Nroot=RootList
    return