import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))))
from sdspark.DMGraphScan.DMGraphScan import GraphScan as dmgraphscan
from sdspark.DepthFirstScan.DFS import GraphScan as dfs
from sdspark.AdditiveScan.AdditiveScan import GraphScan as additivescan
from sdspark.NPHGS.NPHGS import GraphScan as nphgs
from sdspark.Meden.Meden import GraphScan as meden
from sdspark.EventTree.EventTree import GraphScan as eventtree
import re,json,copy
from sdspark.measure import *
from pyspark import SparkContext, SparkConf

sc=None
methods={1:"DMGraphScan",2:"DepthFirstScan",3:"AdditiveScan",4:"NPHGS",5:"Meden",6:"EventTree"}

def sc_start(app):
    global sc
    SparkContext.setSystemProperty("spark.port.maxRetries","100")
    SparkContext.setSystemProperty("spark.ui.enabled","false")
    SparkContext.setSystemProperty("spark.task.cpus","2")
    SparkContext.setSystemProperty("spark.driver.memory","100g")
    SparkContext.setSystemProperty("spark.driver.maxResultSize","20g")
    SparkContext.setSystemProperty("spark.driver.cores","4")
    SparkContext.setSystemProperty("spark.executor.instances","25")
    sc=SparkContext.getOrCreate()

def sc_wrap(func):
    def wrapper(*args,**kwargs):
        sc_start("SubgraphScan")
        ret=func(*args,**kwargs)
        return ret
    return wrapper

@sc_wrap
def genE(Graph):
    global sc
    E=sc.parallelize([(edge["index"],(int(edge["source"]),int(edge["target"]))) for edge in Graph["edges"]])
    Pvalue=sc.parallelize([(edge["index"],edge["value"]) for edge in Graph["edges"]])

    return E,Pvalue

@sc_wrap
def genG(Graph):
    global sc
    edges=set()
    for edge in Graph["edges"]:
        edges.add((int(edge["source"]),int(edge["target"])))
        edges.add((int(edge["target"]),int(edge["source"])))
    graph=sc.parallelize([(edge[0],[edge[1]]) for edge in edges])\
        .reduceByKey(lambda a,b:a+b)
    return graph

@sc_wrap
def genSP(Graph,slice):
    # slice starts from 0
    global sc 
    Pvalue=sc.parallelize([(int(node["name"]),node["value"][slice:slice+1] )for node in Graph["nodes"]]) 
    return Pvalue

@sc_wrap
def genP(Graph):
    global sc 
    Pvalue=sc.parallelize([(int(node["name"]),node["value"] )for node in Graph["nodes"]])
    return Pvalue

#? selected
def writeFile(outroot,method,Graph,result):
    RC=result.collect()
    if isinstance(RC[0],list):
        result_nodes=RC
        print "go Single"
    elif isinstance(RC[0],tuple):
        result_nodes=[each[1] for each in sorted(RC,key=lambda x: x[0])]
        print "go Complex"
    else:
        print "Error: this cannot occur"
        sys.exit()

    global methods
    for slice in range(Graph["slices"]):
        outputG=copy.deepcopy(Graph)
        fw=open(os.path.join(outroot,str(slice)+'_'+methods[method]+'.json'),'w+')
        for node in outputG["nodes"]:
            if int(node["name"]) in result_nodes[slice]:
                node["selected"]=True
            else:
                node["selected"]=False
            node["value"]=node["value"][slice]
        json.dump(outputG,fw)
        fw.close()

if __name__=="__main__":
    lg=logger()
    
    # python SubgraphDetection   data     1 
    #        argv[0]             argv[1]  argv[2]
    froot = os.path.join(sys.argv[1])
    outroot= os.path.join(sys.argv[1],'output')
    if not os.path.exists(outroot):
        os.mkdir(outroot)
    f=open(os.path.join(froot,'Graph.json'),'r')
    Graph=json.load(f) 
    slices=Graph["slices"]
    #slice starts from 0

    method = int(sys.argv[2])
    # 1 DMGraphScan  6 EventTree 
    # 2 DFS  3 Addi  4 NPHGS
    # 5 Meden
    if method in [1,6]:
        Graph_RDD=genG(Graph)
        Pvalue_RDD=genP(Graph)
        if method==1:
            result = dmgraphscan(Graph_RDD,Pvalue_RDD,alpha_max=float(sys.argv[3]),input_B=int(sys.argv[4]))
        elif method==6:
            result = eventtree(Graph_RDD,Pvalue_RDD,alpha_max=float(sys.argv[3]))
        writeFile(outroot,method,Graph,result)
    
    elif method in [2,3,4]:
        Graph_RDD=genG(Graph)
        global sc
        Results=sc.parallelize([])
        for slice in range(0,slices):
            Pvalue_RDD=genSP(Graph,slice)
            if method==2: 
                result= dfs(Graph_RDD,Pvalue_RDD,radius=float(sys.argv[3]),anomaly_ratio=float(sys.argv[4]),minutes=int(sys.argv[5]),alpha_max=float(sys.argv[6]))
            elif method==3:
                result= additivescan(Graph_RDD,Pvalue_RDD,npss=sys.argv[3],iterations_bound=int(sys.argv[4]),ncores=int(sys.argv[5]),minutes=float(sys.argv[6]))
            elif method==4:
                result = nphgs(Graph_RDD, Pvalue_RDD,alpha_max=float(sys.argv[3]),npss=sys.argv[4])
            Results=Results.union(result)
        writeFile(outroot,method,Graph,Results)
    
    elif method in [5]:
        E,Pvalue=genE(Graph)
        if method==5:
            result = meden(E,Pvalue,alpha_max=float(sys.argv[3]))
        writeFile(outroot,method,Graph,result)
    


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
