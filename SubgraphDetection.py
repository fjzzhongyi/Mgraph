import DMGraphScan.DMGraphScan
import DepthFirstScan.DFS
import AdditiveScan.AdditiveScan
import NPHGS.NPHGS
import Meden.Meden 
import EventTree.EventTree
import os,sys,re
from pyspark import SparkContext, SparkConf
from measure import *

sc=None
def sc_start(app):
    global sc
    SparkContext.setSystemProperty("spark.ui.enabled","false")
    sc=SparkContext.getOrCreate()

def sc_wrap(func):
    def wrapper(*args,**kwargs):
        sc_start("SubgraphScan")
        ret=func(*args,**kwargs)
        return ret
    return wrapper

@sc_wrap
def genE(froot):
    """
    # python version
    E=[]
    # f2 a line:  2803301701 3022787727
    f2=open(os.path.join(froot,'E'),'r')
    s=f2.readline()
    while len(s)>0:
        E.append(s.strip().replace(' ','-'))
        s=f2.readline()
    f2.close()
    """
    # SPARK version
    global sc
    text =sc.textFile(os.path.join(froot,'E'))
    E=text.map(lambda x: x.replace(' ','-')).collect()
    
    return E

@sc_wrap
def genG(froot):
    def doublemap(line):
        n1,n2=line.split(' ')
        return [line,n2+' '+n1]
    def linemap(line):
        n1,n2=line.split(' ')
        return (int(n1),[int(n2)])
    # SPARK version
    global sc
    graph=sc.textFile(os.path.join(froot,'G'))\
            .flatMap(doublemap)\
            .distinct()\
            .map(linemap)\
            .reduceByKey(lambda a,b:a+b)
    return graph

@sc_wrap
def genSP(froot,slice):
    # slice starts from 0
    
    def linedec(line):
        global slice
        eles=re.split(" |:",line)
        return (int(eles[0]),[float(eles[slice+1])])

    global sc
    Pvalue=sc.textFile(os.path.join(froot,'P'))\
             .map(linedec)
    return Pvalue

@sc_wrap
def genP(froot):
    def linedec(line):
        global slice
        eles=re.split(" |:",line)
        return (int(eles[0]),[float(ele)  for ele in eles[1:]])

    global sc
    Pvalue=sc.textFile(os.path.join(froot,'P'))\
             .map(linedec)
    return Pvalue

def writeFile4D(outroot,method,result):
    fw=open(os.path.join(outroot,str(method)+'.txt'),'w+')
    for each in result.sortByKey().collect():
        fw.write(' '.join([str(ei) for ei in each[1]]))
        fw.write('\n')
        fw.flush()
    fw.close()
def writeFile4S(outroot,method,result):
    fw=open(os.path.join(outroot,str(method)+'.txt'),'w+')
    for each in result.collect():
        fw.write(' '.join([str(ei) for ei in each]))
        fw.write('\n')
        fw.flush()
    fw.close()

def getSlices(froot):
    f=open(os.path.join(froot,'P'),'r')
    s=f.readline().strip().split(' ')
    f.close()
    return len(s)

if __name__=="__main__":
    lg=logger()
    
    # python SubgraphDetection data 1 
    froot = os.path.join(sys.argv[1])
    outroot= os.path.join(sys.argv[1],'output')
    if not os.path.exists(outroot):
        os.mkdir(outroot)
    slices = getSlices(froot)
    #slice starts from 0

    method = int(sys.argv[2])
    # 1 DMGraphScan  6 EventTree 
    # 2 DFS  3 Addi  4 NPHGS
    # 5 Meden
    if method in [1,6]:
        Graph=genG(froot)
        Pvalue=genP(froot)
        if method==1:
            result = DMGraphScan.DMGraphScan.GraphScan(Graph,Pvalue,verbose=True,input_B=10)
        elif method==6:
            result = EventTree.EventTree.detection(Graph,Pvalue,alpha_max=0.15)
        writeFile4D(outroot,method,result)
    
    elif method in [2,3,4]:
        Graph=genG(froot)
        global sc
        Results=sc.parallelize([])
        for slice in range(0,slices):
            Pvalue=genSP(froot,slice)
            if method==2: 
                result= DepthFirstScan.DFS.depth_first_subgraph_detection(Graph,Pvalue)
            elif method==3:
                result= AdditiveScan.AdditiveScan.GraphScan(Graph,Pvalue,npss='BJ')
            elif method==4:
                result = NPHGS.NPHGS.GraphScan(Graph, Pvalue,alpha_max=0.15)
            Results=Results.union(result)
        writeFile4S(outroot,method,Results)
    
    elif method in [5]:
        E=genE(froot)
        Pvalue=genP(froot)
        if method==5:
            result = DMGraphScan.dp.DMGraphScan(Graph,Pvalue,verbose=True,input_B=10)
        writeFile4D(outroot,method,result)
    


def test():
    Graph = {0: [1,2], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [1,4]}
    #Pvalue = [[0.05], [0.16], [0.16], [0.16], [0.16], [0.05]]
    Pvalue = [[0.05,0.04], [0.16,0.03], [0.16,0.10], [0.16,0.2], [0.16,0.3], [0.05,0.4]]

    result = NPHGS.NPHGS.graphscan(Graph, Pvalue)
    print "\nNPHGS result"
    print result

    result= DepthFirstScan.DFS.depth_first_subgraph_detection(Graph,Pvalue)
    print "\nDFS result"
    print result

    result= EventTree.EventTree.detection(Graph,Pvalue)
    print "\nEventTree result"
    print result,'\n'
    
    result= AdditiveScan.AdditiveScan.additive_graphscan(Graph,Pvalue,'BJ',Pvalue)
    print "\nAdditiveScan result"
    print result

    E = ['0-1','0-2','1-2','2-3','3-4','4-5','2-5','1-6']
    Pvalue=[[0.1,0.2],[0.1,0.1],[0.1,0.4],[0.3,0.1],[0.4,0.4],[0.5,0.1],[0.05,0.1],[0.3,0.14]]
    result = Meden.Meden.detection(E,Pvalue)
    print "\nMeden result"
    print result

    """
    import NetSpot.NetSpot 
    Graph = ['0-1','0-2','1-2','2-3','3-4','4-5','2-5','1-6']
    Pvalue=[[0.1,0.2],[0.1,0.1],[0.1,0.4],[0.3,0.1],[0.4,0.4],[0.5,0.1],[0.05,0.1],[0.3,0.14]]
    result = NetSpot.NetSpot.detection(Graph,Pvalue)
    print "\nNetSpot result"
    print result
    """
