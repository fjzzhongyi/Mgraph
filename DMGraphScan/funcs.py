from vertex import leaf,nonleaf
from node import Node
from ksubgraph import ISTG
import os,datetime
from GV import *

def add_pvalue():
    print 'add_pvalue...'
    f=open(os.path.join(gv.froot,'pvalues.dat'),'r')
    s=f.readline()
    gv.duration=len(s.strip().split(' '))-1
    while len(s)>0:
        s=s.strip().split(' ',1)
        num=int(s[0])
        if num_name[num] in name_node:
            node=name_node[num_name[num]]
            node.pvalue=[]
            s1= s[1].split(' ')
            for each in s1[gv.start-gv.base:gv.end-gv.base]:
                node.pvalue.append(float(each))
        s=f.readline()

def add_parent():
    print 'add_parent...'
    for eachnode in name_node.values():
        for eachchild in eachnode.child:
            eachchild.parent=eachnode

def genG():
    print 'genG...'
    G={}
    # f1 one line: 1000432103
    f1=open(os.path.join(gv.froot,'nodes.dat'),'r')
    # f2 a line:  2803301701 3022787727
    f2=open(os.path.join(gv.froot,'edges.dat'),'r')
    s=f1.readline()
    
    num=0
    while len(s)>0:
        name=s.strip()
        num_name[num]=name
        G[name]={}
        s=f1.readline()
        num+=1
    
    s=f2.readline()
    while len(s)>0:
        s=s.strip().split(' ')
        G[s[0]][s[1]]=1
        G[s[1]][s[0]]=1
        s=f2.readline()
    f1.close()
    f2.close()
    return G

def connectedG(G):
    print 'connectedG...'
    print len(G)
    keys=G.keys()
    for vroot in keys:
        # s1 set of connected nodes 
        s1=set([vroot])
        # s2 set of nodes to be visited
        s2=set(G[vroot].keys())
        while len(s2)>0:
            # s2 labeled visited
            s1|=s2
            # use neiboughers to modify s2
            s2=set()
            for n in s2:
                s2|=set(G[n].keys()) 
            # only nodes that haven't been visited could be reserved
            s2-=s1
            # if no more nodes are added, break the loop
          
        if len(s1)>=1.0/2.0*len(G):
            print len(s1)
            # modify G
            # s2 are nodes that are to be del
            s2=set(G.keys())-s1

            for v in list(s2):
                for u in G[v].keys():
                    G[u].pop(v)
                G.pop(v)
            break
    return G

def saveISTG(root):
    print ('saveISTG...')
    fw=open(os.path.join(gv.fmroot,'ISTG.txt'),'w+')
    G={}
    def se_dp(n):
        if n.name not in G:
            G[n.name]={}
        for each in n.child:
            G[n.name][each.name]=1
            se_dp(each)
          
    se_dp(root)
    
    fw.write(root.name+'\n')
    for each in G.keys():
        fw.write(each)
        for item in G[each].keys():
            fw.write(','+item)
        fw.write('\n')
        fw.flush()

    fw.close()

def readISTG():
    print ('readISTG...')
    fr=open(os.path.join(gv.fmroot,'ISTG.txt'),'r')
    root=fr.readline().strip()
    
    s=fr.readline()
    while len(s)>0:
        s=s.strip().split(',')
        #if len(s)>=100:
        #    print 'name:'+s[0]+' num:'+str(len(s)-1)
        nodes=[]
        for it in s:
            if it in name_node:
                nodes.append(name_node[it])
            else:
                nodes.append(Node(it,None,0,[]))
        for no in nodes[1:]:
            nodes[0].addChild(no)
            if nodes[0] in no.child:
                print 'ERROR ERROR'
        s=fr.readline()
 
    def test_tree (root):
        class TreeError (RuntimeError):
            def __init__(self,arg):
                self.args=arg
        def t_dp(node):
            if node in nodes:
                raise TreeError('A')
            else:
                nodes.add(node)
            for n in node.child:
                t_dp(n)
        nodes=set([])
        t_dp(root)
    test_tree(name_node[root])


    return name_node[root]

def genGpv():
    print 'genGpv...'
    Gpv={}

    # read from pvalues.dat 
    # f one line: 17 1.00000 1.00000 0 0 1.00000
    f=open(os.path.join(gv.froot,'pvalues.dat'),'r')
    s=f.readline()
    vr=None
    while len(s)>0:
        s=s.strip().split(' ')
        num=int(s[0])
        #pvalue=int(s[1])
        #now we extract the pvalues from s[1] to s[-1]
        # if one element of set of pvalues is anomalous, this node's pvalue will be marked as 0, otherwise 1
        # method: fliter s and reserve those whose value is less than alpha
        if len(filter(lambda x: True if float(x)<=alpha else False, s[1:-1]))>0:
            pvalue = 0
        else:
            pvalue = 1
        if vr is None and pvalue<=alpha:
            vr=num_name[num]
        Gpv[num_name[num]]=pvalue
        s=f.readline()
    return Gpv,vr

def writeOmega(O):
    # here the structure of O is different from omega of Node
    fw=open(os.path.join(gv.fwroot,'result_B='+str(gv.B)+'segment_'+str(gv.segment)+datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')+'_filter'+str(gv.if_filter)+'.txt'),'w+')
    day=1
    for each in O:
        print each
        # assuming that each[0][t] is set of nodes detected. But it contains normal nodes.
        # so now we filter it
        for t in range(len(each)):
            fw.write(str( [ e.name for e in each[t]])+'\n')
            
            # it's wrong to filter nodes now, we should do it before
            #fw.write(str( [ e.name for e in filter(lambda x:x.pvalue<=alpha,each[0][t])] )+'\n')
            
            #fw.write('common attributes\n')
            #fw.write(str( list(each[1][t]) )+'\n\n')
            fw.flush()
    fw.close()
