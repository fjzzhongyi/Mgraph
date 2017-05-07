from vertex import leaf,nonleaf
from node import Node
from ksubgraph import ISTG
import os,datetime,threading,itertools,sys
from GV import *
from funcs import *
from pyspark import SparkContext
#
sc=None
def sc_start():
    global sc
    sc=SparkContext.getOrCreate()

def dGraphScan(root):
    def  dp_r(node):
        #recursively
        if len(node.child)>0:
            for child in node.child:
                dp_r(child)
            nonleaf(node)
        else:
            leaf(node)
            return
    def dp_l(root):
        # loop
        #count the number of nodes visited
        #count=0, count = 1/2 * loop, redundant
        loop=0
        node=root
        #child_num is the index of child to be visited
        child_num=0
        stack=[]
        while True:
            loop+=1
            if loop %1000==0:
                print str(loop)+'times'
                print datetime.datetime.now()
            if len(node.child)>0:
                if child_num >=len(node.child):
                    nonleaf(node)
                    # finish visit
                    if len(stack)==0:
                        break
                    #count+=1
                    #if count%1000==0:
                    #    print str(count)+'visited'
                        
                    node,child_num=stack.pop()
                else:
                    stack.append([node,child_num+1])
                    node=node.child[child_num]
                    child_num=0
                    continue
            else:
                leaf(node)
                # finish visit
                if len(stack)==0:
                    break
                #count+=1
                #if count%1000==0:
                #    print count
                #    print datetime.datetime.now()
                node,child_num=stack.pop()
    print 'dGraphScan...'
    
    #{

    dp_l(root)
    #}
    """
    
    # to utilize thread, use recursive dp
    if len(root.child)>0:
        threads=[]
        for child in root.child:
            threads.append(threading.Thread(target=dp_r,args=(child,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        nonleaf(root)
    else:
        leaf(root)
        return
    """
    
    #root.showOmega()
    #root.writeOmega()




def test():
    #name, pvalue, nchild, child
    E=Node('E',[0.02,0.10,0.23],0,[])
    D=Node('D',[0.10,0.30,0.19],0,[])
    C=Node('C',[0.01,0.06,0.12],2,[D,E])
    B=Node('B',[0.19,0.01,0.11],1,[C])

    E.pdvalue=[[0.1,0.11,0.4,0.3],[0.1,0.11,0.11,0.3],[0.3,0.31,0.11,0.3]]
    D.pdvalue=[[0.2,0.1,0.1,0.05],[0.2,0.4,0.2,0.05],[0.2,0.4,0.2,0.35]]
    C.pdvalue=[[0.01,0.11,0.1,0.3],[0.01,0.31,0.1,0.11],[0.01,0.11,0.3,0.11]]
    B.pdvalue=[[0.1,0.3,0.2,0.5],[0.1,0.06,0.12,0.5],[0.1,0.06,0.12,0.1]]
    E.parent=C
    D.parent=C
    C.parent=B

    dGraphScan(B)
#test()



def dp(Graph=None,Pvalue=None,fileinput=True,verbose=False):
    if fileinput is True:
        # we need to add gv. root filepaths
        gv.add_rootdir(sys.argv[1])
    
    
    name_node.clear()
    num_name.clear()
    
    tb=datetime.datetime.now()
    print 'Starting\n'

    if True:
        G=genG(Graph,fileinput)
        G=connectedG(G)
        Gpv,vroot=genGpv(Graph,Pvalue,fileinput)
        print "root is :" + str(vroot)
        # def ISTG(G, Gpv, vroot, galpha)
        root=ISTG(G,Gpv,vroot,alpha)
        #saveISTG(root)
    else:
        G=genG(Graph,fileinput)
        G=connectedG(G)
        Gpv,vroot=genGpv(Pvalue,fileinput)
        root=readISTG()
    print root, root.name, root.nchild, root.child, root.parent
    add_parent()

    daysOmega=[]
    daysOmega2=[]
    nnn=0

    #gv.start=42
    #gv.end=43
    #nnn+=1
    #gv.slices=gv.end-gv.start 
    ##add pvalue, pdvalue
    #add_pvalue()
    #add_pdvalue()
    #dGraphScan(root)
    #daysOmega.append(root.omega[-1])
    getDuration(Pvalue,fileinput)
    while True:
        gv.start=gv.base+gv.segment*nnn
        gv.end=min([gv.base+gv.segment*(nnn+1),gv.base+gv.duration])
        nnn+=1
        gv.slices=gv.end-gv.start
        #add pvalue, pdvalu
        add_pvalue(Pvalue,fileinput)
        dGraphScan(root)
        # -1 denotes that this omega is consistent to the biggest B, which always performs best
        daysOmega.append(root.omega[-1])
        daysOmega2.append(root.getFilterOmega())
        if gv.end==gv.base+gv.duration:
            break
    
    te=datetime.datetime.now()
    print 'Time Duration:'+ str((te-tb).seconds)
    # output result to files or return 
    if fileinput is True:
        gv.if_filter=False
        writeOmega(daysOmega)
        gv.if_filter=True
        writeOmega(daysOmega2)
    else:
        # assuming that this program is not run in segments
        return [[int(node.name) for node in each] for each in root.omega[-1]]

def sc_wrap(func):
    def wrapper(*args,**kwargs):
        sc_start()
        ret=func(*args,**kwargs)
        return ret
    return wrapper

def RDDdec(Graph_RDD,Pvalue_RDD):

#Graph   FOR nodewise: connections (nodeNo., [connected nodes]) / (int, [int,int,...])
#       FOR edgewise: (edgeNo. , (nodeA, nodeB))
#Pvalue  (node,valuelist) / (int,[float,...])
#
#Graph   FOR:  nodewise   {0:[],1:[],2:[]} node start from zero 0
#        FOR:  edgewise   [edge1, edge2, edge3, ...]     edge%n: str   format like  "1-2"
#Pvalue = [[],[],[],[],[]] which is corresponding with node by index
    Graph={}
    for ele in Graph_RDD.collect():
        Graph[ele[0]]=ele[1]
    Pvalue=\
    [ele[1] for ele in sorted(Pvalue_RDD.collect(),key=lambda x: x[0])]
    print Graph,Pvalue
    return Graph,Pvalue
def SubgraphEnc(subgraphs):
    global sc
    return sc.parallelize([(index,subgraph)for index,subgraph in enumerate(subgraphs)])

@sc_wrap
def GraphScan(Graph_RDD,Pvalue_RDD,alpha_max=0.15,input_B=2,verbose=False):
    Graph,Pvalue=RDDdec(Graph_RDD,Pvalue_RDD)
    gv.B=input_B
    # segment should not play its role in this Function, supposing that input size is not so huge. so no need for cut them into segments
    
    # !!!! alpha is constant, needed modification
    gv.segment=len(Pvalue[0])
    return SubgraphEnc(dp(Graph,Pvalue,fileinput=False,verbose=verbose))


if __name__=='__main__':

    for (B,segment) in itertools.product(BList,segmentList):
        gv.B=B
        gv.segment=segment
        dp()
