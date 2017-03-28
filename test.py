import DMGraphScan.dp
import DepthFirstScan.DFS
import AdditiveScan.AdditiveScan_Traffic

Graph = {0: [1,2], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [1,4]}
#Pvalue = [[0.05], [0.16], [0.16], [0.16], [0.16], [0.05]]
Pvalue = [[0.05,0.04], [0.16,0.03], [0.16,0.10], [0.16,0.2], [0.16,0.3], [0.05,0.4]]

result = DMGraphScan.dp.DMGraphScan(Graph,Pvalue,verbose=True,input_B=10)
print "DMGraphScan result"
print result

result= DepthFirstScan.DFS.depth_first_subgraph_detection(Graph,Pvalue)
print "DFS result"
print result

result= AdditiveScan.AdditiveScan_Traffic.additive_graphscan(Graph,Pvalue,'BJ',Pvalue)
print "AdditiveScan result"
print result

import Meden.Meden 
Graph = ['0-1','0-2','1-2','2-3','3-4','4-5','2-5','1-6']
Pvalue=[[0.1,0.2],[0.1,0.1],[0.1,0.4],[0.3,0.1],[0.4,0.4],[0.5,0.1],[0.05,0.1],[0.3,0.14]]
result = Meden.Meden.detection(Graph,Pvalue)
print "Meden result"
print result
