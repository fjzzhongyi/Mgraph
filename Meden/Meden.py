import commands
import sys,os,re
def GraphScan(Graph,Pvalue,alpha_max=0.15):
    #Graph [edge1, edge2, edge3, ]
    #         edge%n: str   format like  "1_2"
    #Pvalue [[float, ],[],[],[]...]

    # to be finished: transform vertex_graph which only vertices have pvalues into edge_graph which only edges have pvalues   
    """    
    # transform Vertex_Graph into Edge_Graph
    # Edge marked as Vertex
    edge2vertex={}
    ori_num=0
    for node1, n2list in Graph.items():
        for node2 in n2list:
            if (node1,node2) not in edge2vertex and (node2,node1) not in edge2vertex:
                edge2vertex[(node1,node2)]=ori
                ori_num+=1
    # ori_num = num of nodes after transformation
    vertex2edge={v:k for k,v in edge2vertex.items()}

    # iterative operations upon old vertices
    # if one edge is adjacent to another edge, now the two corresponding vertices should have one new edge connecting them 
    for old_node in Graph.keys():
    """
    # create file 
    # I give up trying to modify the path, let it stay where the calling module stays
    # filename=os.path.split(this.__file__)[0]+os.sep+"temp.txt"
    filepath= os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    temppath=filepath+os.sep+"temp.txt"
    medenpath=filepath+os.sep+"meden.jar"
    start=0
    end=len(Pvalue[0])-1
    with open (temppath,'w+') as fw:
        for t in range(len(Pvalue[0])):
            # assuming edge format node1_node2 like 1-2, type str
            for edge_index in range(len(Graph)):
                fw.write(Graph[edge_index].replace('-',','))
                fw.write(','+str(t)+','+str(Pvalue[edge_index][t])+'\n')
    command = "java -Xms1024m -Xmx2048m -jar "+medenpath+" -run "+temppath+" "+str(start)+" "+str(end)+" p "+str(alpha_max)
    line= str(commands.getoutput(command))
    os.remove(temppath) 
    
    #result is set of edges going along with t-slice:  [[edge1,edge2,]...]
    result = [[]for i in range(len(Pvalue[0]))]
    outputs= line.split('\n')
    print outputs
    t1,t2=outputs[2].split(']')[0].split('[')[1].split(',')
    p=re.compile(r' \d*-\d*\(-?.*?\)')
    for t in range(int(t1),int(t2)+1):
        items=p.findall(re.search(r'\{.*\}',outputs[3+t-int(t1)]).group(0)) 
        # item example : " 88-47(0.32)"
        print items
        for item in items:
            result[t].append(item.lstrip(' ').split('(')[0])
    return result
