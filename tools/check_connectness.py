import json,os

def find_max_connected(Graph):
    S=set(Graph.keys())
    subgraphs=[]
    while len(S)>0:
        M=S.pop()
        EXT=set(Graph[M])
        M=set([M])
        while len(EXT)>0:
            Next=set([n2  for n1 in EXT for n2 in Graph[n1]])
            M|=EXT
            EXT=Next-M
        S-=M
        subgraphs.append(M)
    return sorted(subgraphs,key=lambda x: -1*len(x))[0]

def operate(inputfile,outputfile):
    if not os.path.exists(inputfile):
        raise IOError("your input file not exists, please check it carefully")

    graph=json.load(open(inputfile,'r'))
    G=set()
    Graph={}
    for edge in graph["edges"]:
        n1,n2=int(edge["source"]),int(edge["target"])
        G.add(n1)
        if n1 in Graph:
            Graph[n1].append(n2)
        else:
            Graph[n1]=[n2]
        G.add(n2)
        if n2 in Graph:
            Graph[n2].append(n1)
        else:
            Graph[n2]=[n1]
    connected_nodes= set(find_max_connected(Graph))
    print "initial nodes' number:",len(G)
    print "connected nodes' number:",len(connected_nodes)
    G=sorted(list(G&connected_nodes))
    
    i=0
    while i<len(graph["nodes"]):
        node=graph["nodes"][i]
        if int(node["name"]) not in G:
            graph["nodes"].pop(i)
        else:
            i+=1
    i=0
    while i<len(graph["edges"]):
        edge=graph["edges"][i]
        if int(edge["source"]) not in G and int(edge["target"])not in G:
            graph["edges"].pop(i)
        else:
            i+=1
    json.dump(graph,open(outputfile,'w+'))
    print "Successfully find out maximal connected components"

if __name__=='__main__':
    operate('Graph.json','newg.json')


"""
class UnionSet(object):
    def __init__(self,graph):
        self.parent = {}
        self.graph=graph
    def init(self, key):
        if key not in self.parent:
            self.parent[key] = key
            
    def find(self, key):
        self.init(key)
        while self.parent[key] != key:  
            self.parent[key] = self.parent[self.parent[key]] 
            key = self.parent[key]
        return key      

    def join(self, key1, key2):
        p1 = self.find(key1)
        p2 = self.find(key2)
        if p1 != p2:
            self.parent[p2] = p1
    def find_max_connected(self):
        edges=[]
        for n1 in self.graph:
            for n2 in self.graph[n1]:
                if (n1,n2) not in edges and (n2,n1) not in edges:
                    edges.append((n1,n2))
        #for (n1,n2) in edges:        
        #    self.init2(n1,n2)
        print len(set(self.parent.values()))
        for (n1,n2) in edges:
            self.join(n1,n2)
        print len(set(self.parent.values()))
        
        subgraphs={}
        
        for key,value in self.parent.items():
            if value not in subgraphs:
                subgraphs[value]=[key]
            else:
                subgraphs[value].append(key)
        return sorted(subgraphs.values(),key=lambda x: -1*len(x))[0]
"""
