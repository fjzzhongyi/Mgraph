import commands
import sys,os,re
from pyspark import SparkContext

sc=None
def sc_start():
    global sc
    sc=SparkContext.getOrCreate()

def sc_wrap(func):
    def wrapper(*args,**kwargs):
        sc_start()
        ret=func(*args,**kwargs)
        return ret
    return wrapper
def RDDdec(Graph_RDD,Pvalue_RDD):
    Graph={}
    for ele in Graph_RDD.collect():
        Graph[ele[0]]=ele[1]
    Pvalue=\
    [ele[1] for ele in sorted(Pvalue_RDD.collect(),key=lambda x: x[0])]
    return Graph,Pvalue
def SubgraphEnc(subgraphs):
    global sc
    if len(subgraphs)==0 or not isinstance(subgraphs[0],list):
        reRDD=sc.parallelize([subgraphs])
    else:
        reRDD=sc.parallelize([(index,subgraph)for index,subgraph in enumerate(subgraphs)])
    return reRDD
@sc_wrap
def GraphScan(Graph_RDD,Pvalue_RDD,alpha_max=0.15):
    Graph,Pvalue=RDDdec(Graph_RDD,Pvalue_RDD)
    #Graph {0:[1,2],...}
    #Pvalue=[[],[],[]]
    filepath= os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    Gpath=filepath+os.sep+"G"
    Ppath=filepath+os.sep+"P"
    Rpath=filepath+os.sep+"RESULT"
    EventtreePath=filepath+os.sep+"EventTree.jar"
    with open (Gpath,'w+') as fw1 ,open (Ppath,'w+')as fw2:
        for key,value in Graph.items():
            for n2 in value:
                fw1.write(str(key)+' '+str(n2)+'\n')
        for ele in Pvalue:
            fw2.write(' '.join([str(v) for v in ele ])+'\n')
    command= "java -jar "+EventtreePath+" "+str(alpha_max)+" "+filepath
    line = str(commands.getoutput(command))
    print line 
    with open(Rpath,'r') as fr1:
        result=[[int(ele) for ele in  line.split(" ")]   for line in fr1.readlines()]

    os.remove(Rpath)
    os.remove(Gpath)
    os.remove(Ppath)
    return SubgraphEnc(result)
