import os,sys,datetime
import random
import json
import math
from os.path import join
from sets import Set
import sys
import numpy as np
import copy
import time
import npssScore as npssS
import networkx as nx

name_node={}
num_name={}

def detection(PValue, E, alpha_max='0.15', npss='BJ', verbose_level = 0):#pvalue, network,alpha

	if verbose_level == 2:
		print 'PValue : ',PValue # is a dictionary, i:p-valuei
		print 'E : ',E # is a dictionary, 'i_j':1.0
		print 'alpha_max : ',alpha_max
		print 'npss : ',npss

	graph = {}
	for item in E:
		vertices = item.split('_')
		i = int(vertices[0])
		j = int(vertices[1])
		if i not in graph:
			graph[i] = {}
		graph[i][j] = E[item]
		if j not in graph:
			graph[j] = {}
		graph[j][i] = E[item]
	
	V = []
	S_STAR = []
	for nid, pv in PValue.items(): V.append([nid, pv])
	V = sorted(V, key=lambda xx:xx[1])#all node+pvalue base on p value decide priority node
	'''
	graph is a adjacency list id: dictionary
	'''
	if verbose_level == 2:
		for index,value in graph.items():
			print 'index : ',index , 'adj : ',value
		for item in V:
			print item 
	candidate = [k[0] for k in V if k[1] <= alpha_max]
	
	count = 1
	for item in V:
		if item[1] <= 0.15: count = count + 1
		else : break
	K = count
	print count
	for k in range(K):
		initNodeID = V[k][0]
		initNodePvalue = V[k][1] 
		S = {initNodeID:initNodePvalue} #{nid:pvalue}
		max_npss_score = npssS.npss_score([[item, S[item]] for item in S], alpha_max, npss)
		maxS = [[initNodeID, initNodePvalue]]
		while(True):
			G = []
			for v1 in S:
				if v1 not in graph:
					print v1
					print S
					print 'could not happen ...'
					sys.exit()
					continue
				for v2 in graph[v1]:
					if v2 not in S and v2 not in [item[0] for item in G]:
						G.append( [  v2, PValue[v2]  ] )
			G = sorted(G, key=lambda xx:xx[1])
			S1 = [[item, S[item]] for item in S]
			for item in G:
				S1.append(item)
				phi = npssS.npss_score(S1, alpha_max, npss)
				if phi > max_npss_score:
					maxS = copy.deepcopy(S1)
					max_npss_score = phi
			if len(maxS) == len(S) :
				break
			else:
				S = {item[0]:item[1] for item in maxS}
			if verbose_level != 0:
				print 'len(S) : ',len(S), ' len(maxS) : ',len(maxS)
				print 'S : ',[item for item in S]
				print 'maxS : ',[item[0] for item in maxS]
		S_STAR.append([sorted(maxS, key = lambda item: item[0]), max_npss_score])# subgraphs and score
	S_STAR = sorted(S_STAR, key=lambda xx:xx[1])
	if verbose_level != 0:
		for item in S_STAR:
			print item
	'''return the maximum npss value '''
	if len(S_STAR) > 0:
		subGraph  = S_STAR[len(S_STAR)-1]
		return [item[0] for item in subGraph[0]] , subGraph[1]
	else:
		print 'kddgreedy fails, check the error'
		sys.exit()
		return None
	
def genG(froot):
	print 'genG...'
	G={}
	# f1 one line: 1000432103
	f1=open(os.path.join(froot,'nodes.dat'),'r')
	# f2 a line:  2803301701 3022787727
	f2=open(os.path.join(froot,'edges.dat'),'r')
	
	line=0
	s=f1.readline()
	while len(s)>0:
		s=s.strip()
		num_name[line]=int(s)
		line+=1
		s=f1.readline()
		
	s=f2.readline()
	while len(s)>0:
		s=s.strip().split(' ')
		G[s[0]+'_'+s[1]]=1.0
		s=f2.readline()
	f1.close()
	f2.close()
	return G

def addPvalue(froot,slice):
	print 'add_pvalue...'
	Pvalue={}
	f=open(os.path.join(froot,'pvalues.dat'),'r')
	s=f.readline()
	while len(s)>0:
		s=s.strip().split(' ')
		Pvalue[num_name[int(s[0])]]=float(s[slice])
		s=f.readline()
	#print Pvalue
	return Pvalue
	
def getSlices(froot):
	print 'getSlices'
	f=open(os.path.join(froot,'pvalues.dat'),'r')
	s=f.readline().strip().split(' ')
	print 'slices= '+str(len(s)-1)
	return len(s)
	
		
if __name__ =='__main__':
	froot = os.path.join(sys.argv[1],'input')
	outroot= os.path.join(sys.argv[1],'output')
	alpha_max=0.15
	npss='BJ'
	G=genG(froot)
	slices=getSlices(froot)
	result=[]
	
	startTime = time.time()
	print 'start processing : '
	for slice in range(1,slices):
		print 'slice=' +str(slice)+'...'
		Pvalue=addPvalue(froot,slice)
		resultNodes,score =detection(Pvalue,G,alpha_max,npss)
		result.append((resultNodes,score))
	runningTime = time.time() - startTime
	print 'finishing ,duration: '+ str(runningTime)
	
	fw=open('NPHGS_'+datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')+'.txt','w+')
	for each in result:
		resultNodes=each[0]
		score=each[1]
		fw.write(str([str(each) for each in resultNodes]))
		fw.write('\n')
		fw.flush()
	fw.close()
