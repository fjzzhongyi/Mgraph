from GV import name_node,gv,alpha
#predefine gv.slices, B

"""
class Node:
    def __init__(self, name, pvalue, nchild, child):
        self.name = name
        self.pvalue = pvalue
        self.nchild = nchild
        self.child = child
"""


class Node:

    def __init__(self, name, pvalue, nchild, child):
        self.name=name
        self.parent=None
        self.child=child
        self.nchild=nchild
        #if len(pvalue)!=gv.slices:
        #    print 'wrong1'
        self.pvalue=pvalue
        
        # modify part of omega  from  *set()*  to   *( set, set )*
        # new: self.omega=[[[set() for i in range(GV.gv.slices)] for j in range(2)] for j in range(GV.B+1)]
        # b, 0/1, t
        # changed to self.omega=[[set() for i in range(GV.gv.slices)] for j in range(GV.B+1)]
        self.omega=None
        # b
        self.BJ=[0 for j in range(gv.B+1)]
        # b, t
        self.Nroot=None

        # omegaN denotes psi
        #self.omegaN=None

        name_node[name]=self
    def getFilterOmega(self):
        for t in range(len(self.omega[-1])):
            self.omega[-1][t]= set( filter(lambda x: x.pvalue[t]<=alpha ,self.omega[-1][t]) )
        return self.omega[-1]

    def addChild(self,child):
        self.child.append(child)
        self.nchild+=1
        child.parent=self
    def showOmega(self):
        print "Omega of "+ self.name +" is:"
        for b in range(gv.B,gv.B+1):
            print 'B=' + str(b) 
            print '\n\nnodes'
            print [[ each.name for each in self.omega[b][t]] for t in range(gv.slices)]
        print '\n\n\n\n'
    def writeOmega(self):
        fw=open('result.txt','a')
        fw.write( "Omega of "+ self.name +" is:\n")
        for b in range(gv.B,gv.B+1):
            fw.write( 'B=' + str(b)+'\n') 
            fw.write( '\n\nnodes\n')
            fw.write(str( [[ each.name for each in self.omega[b][t]] for t in range(gv.slices)])+'\n')
        fw.write( '\n\n\n\n')
        fw.close()
