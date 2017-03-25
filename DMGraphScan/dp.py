from vertex import leaf,nonleaf
from node import Node
from ksubgraph import ISTG
import os,datetime,threading,itertools
from GV import *
from funcs import *
#


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



def dp():
    name_node.clear()
    num_name.clear()
    
    tb=datetime.datetime.now()
    print 'Starting\n'

    if True:
        G=genG()
        G=connectedG(G)
        Gpv,vroot=genGpv()
        print vroot
        # def ISTG(G, Gpv, vroot, galpha)
        root=ISTG(G,Gpv,vroot,alpha)
        saveISTG(root)
    else:
        G=genG()
        G=connectedG(G)
        Gpv,vroot=genGpv()
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


    while True:
        gv.start=gv.base+gv.segment*nnn
        gv.end=min([gv.base+gv.segment*(nnn+1),gv.base+gv.duration])
        nnn+=1
        gv.slices=gv.end-gv.start 
        #add pvalue, pdvalu
        add_pvalue()
        dGraphScan(root)
        daysOmega.append(root.omega[-1])
        daysOmega2.append(root.getFilterOmega())
        if gv.end==gv.base+gv.duration:
            break
    
    gv.if_filter=False
    writeOmega(daysOmega)
    gv.if_filter=True
    writeOmega(daysOmega2)
    
    te=datetime.datetime.now()
    print 'Duration:'+ str((te-tb).seconds)

if __name__=='__main__':

    for (B,segment) in itertools.product(BList,segmentList):
        gv.B=B
        gv.segment=segment
        dp()
