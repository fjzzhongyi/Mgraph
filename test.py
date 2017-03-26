import DMGraphScan.dp
G = {0: [1,2], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [1,4]}
#Pvalue = [[0.05], [0.16], [0.16], [0.16], [0.16], [0.05]]
Pvalue = [[0.05,0.04], [0.16,0.03], [0.16,0.10], [0.16,0.2], [0.16,0.3], [0.05,0.4]]
print DMGraphScan.dp.DMGraphScan(G,Pvalue,verbose=True,input_B=10)
