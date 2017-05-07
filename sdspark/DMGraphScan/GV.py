#Global Variables
import os,sys
#haze:272  5~276 [5:277] 2014.04.15~2015.1.11
#flu:225  2~226 [2:227]  week2~ week226

# segment is different from slices, satisfying condition that segment>=slices
# segment is static and slices is decided by start and end
BList=range(0,1)
segmentList=range(1,8)

solutionlists={}

class Glov:
    def __init__(self):
        self.base=1
        self.duration=0
        self.slices=-1
        self.start=-1
        self.end=-1
        self.segment=2
        self.B=2
        self.if_filter=False
    
    def add_rootdir(self,rootdir):     
        self.froot=os.path.join(rootdir,'input')
        self.fwroot=os.path.join(rootdir,'output')
        self.fmroot=os.path.join(rootdir,'meta')
        if not os.path.exists(self.fwroot):
            os.makedirs(self.fwroot)
        if not os.path.exists(self.fmroot):
            os.makedirs(self.fmroot)
gv=Glov()
########
########
alpha=0.15
name_node={}
num_name={}

