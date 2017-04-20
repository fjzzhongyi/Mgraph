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

def operate(f1,f2):
    with open(f1,'r') as fr1, open(f2,'r') as fr2:
        f1_lines=fr1.readlines()
        f2_lines=fr2.readlines()
        with open('G','w+') as fw1, open('P','w+') as fw2:
            G=set()
            Graph={}
            for s in f1_lines:
                n1,n2=[int (i) for i in s.strip().split(' ')]
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
            print len(connected_nodes)
            print len(G)
            G=sorted(list(G&connected_nodes))
            times=0
            for s in f1_lines:
                fw1.write(' '.join([str(G.index(int(ele))) for ele in s.strip().split(' ')])+'\n')
                times+=1
            """
              
            sc.broadcast(G)
            text=sc.textFile('G')
            for each in text.map(lambda s:' '.join([str(G.index(int(ele))) for ele in s.strip().split(' ')])+'\n').collect():
                fw1.write(each)
            print "start"
            """
            for index,s in enumerate(f2_lines):
                if index in G:
                    fw2.write(s)

if __name__=='__main__':
    operate('G1','P1')
