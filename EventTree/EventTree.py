import commands
import sys,os,re

def GraphScan(Graph,Pvalue,alpha_max=0.15):
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
    return result
