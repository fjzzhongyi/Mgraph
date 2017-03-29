import DMGraphScan.dp
import DepthFirstScan.DFS
import AdditiveScan.AdditiveScan
import NPHGS.NPHGS
import Meden.Meden 
import os,sys

def genE(froot):
	E=[]
	# f2 a line:  2803301701 3022787727
	f2=open(os.path.join(froot,'E'),'r')
		
	s=f2.readline()
	while len(s)>0:
		E.append(s.strip().replace(' ','-'))
        s=f2.readline()
	f2.close()
	return E

def genG(froot):
	graph={}
	# f2 a line:  2803301701 3022787727
	f2=open(os.path.join(froot,'G'),'r')
		
	s=f2.readline()
	while len(s)>0:
		s=s.strip().split(' ')
		n1=int(s[0])
		n2=int(s[1])
		if graph.has_key(n1):
			if n2 not in graph[n1]:
				graph[n1].append(n2)
		else:
			graph[n1] = [n2]
		if graph.has_key(n2):
			if n1 not in graph[n2]:
				graph[n2].append(n1)
		else:
			graph[n2] = [n1]
		s=f2.readline()
	f2.close()
	return graph

def genSP(froot,slice):
    # slice starts from 0
    Pvalue=[]
    f=open(os.path.join(froot,'P'),'r')
    s=f.readline()
    while len(s)>0:
        s=s.strip().split(' ')
        Pvalue.append([float(s[slice])])
        s=f.readline()
    return Pvalue

def genP(froot):
    Pvalue=[]
    f=open(os.path.join(froot,'P'),'r')
    s=f.readline()
    while len(s)>0:
        s=s.strip().split(' ')
        Pvalue.append([float(i) for i in s])
        s=f.readline()
    return Pvalue

def writeFile(outroot,method,result):
    fw=open(os.path.join(outroot,str(method)+'.txt'),'w+')
    for each in result:
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
    # python SubgraphDetection data 1 
    froot = os.path.join(sys.argv[1])
    outroot= os.path.join(sys.argv[1],'output')
    if not os.path.exists(outroot):
        os.mkdir(outroot)
    slices = getSlices(froot)
    #slice starts from 0

    method = int(sys.argv[2])
    # 1 DMGraphScan 
    # 2 3 4 DFS Addi NPHGS
    # 5 Meden
    if method in [1]:
        Graph=genG(froot)
        Pvalue=genP(froot)
        if method==1:
            result = DMGraphScan.dp.DMGraphScan(Graph,Pvalue,verbose=True,input_B=10)
        writeFile(outroot,method,result)
    
    elif method in [2,3,4]:
        Graph=genG(froot)
        Results=[]
        for slice in range(0,slices):
            Pvalue=genSP(froot,slice)
            if method==2: 
                result= DepthFirstScan.DFS.depth_first_subgraph_detection(Graph,Pvalue)
            elif method==3:
                result= AdditiveScan.AdditiveScan.additive_graphscan(Graph,Pvalue,'BJ',Pvalue)
            elif method==4:
                result = NPHGS.NPHGS.graphscan(Graph, Pvalue)
            Results.append(result[0])
        writeFile(outroot,method,Results)
    
    elif method in [5]:
        E=genE(froot)
        Pvalue=genP(froot)
        if method==5:
            result = DMGraphScan.dp.DMGraphScan(Graph,Pvalue,verbose=True,input_B=10)
        writeFile(outroot,method,result)
        

def test():
    Graph = {0: [1,2], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [1,4]}
    #Pvalue = [[0.05], [0.16], [0.16], [0.16], [0.16], [0.05]]
    Pvalue = [[0.05,0.04], [0.16,0.03], [0.16,0.10], [0.16,0.2], [0.16,0.3], [0.05,0.4]]

    result = NPHGS.NPHGS.graphscan(Graph, Pvalue)
    print "\nNPHGS result"
    print result

    result = DMGraphScan.dp.DMGraphScan(Graph,Pvalue,verbose=True,input_B=10)
    print "\nDMGraphScan result"
    print result

    result= DepthFirstScan.DFS.depth_first_subgraph_detection(Graph,Pvalue)
    print "\nDFS result"
    print result

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
