from GV import solutionlists
import threading,datetime
"""
pieces=3, number = 2
(0,0,2)
(0,1,1)
(1,0,1)
(1,1,0)
(0,2,0)
(2,0,0)
"""

def genlist1 (pieces, number): 
    if (pieces,number) in solutionlists:
        return solutionlists[(pieces,number)]
    res=[]
    def recur(level, number, lis, pieces):
        if level>pieces:
            if  number==0:   
                res.append(lis)
            return
        if level==1:
            threads=[]
            for i in range(number,-1,-1):
                threads.append(threading.Thread(target=recur,args=(level+1, number-i, lis+[i], pieces,)))
            for th in threads:
                th.start()
            for th in threads:
                th.join()
        else:
            for i in range(number,-1,-1):
                recur(level+1, number-i, lis+[i], pieces)
    recur(1, number, [], pieces)
    solutionlists[(pieces,number)]=res
    return res
def genlist (pie, num): 
    if (pie,num) in solutionlists:
        return solutionlists[(pie,num)]    
    res=[]
    level,number,lis,pieces,ic = 1, num, [], pie, num
    stack=[]
    while True:
        if level>pieces:
            if  number==0:   
                res.append(lis)
            if len(stack)==0:
                break
            level,number,lis,pieces,ic= stack.pop()
            continue
        if ic>=0:
            
            stack.append([level, number, lis+[], pieces, ic-1])
            level+=1
            number-=ic
            lis+=[ic]
            ic=number
        else:
            if len(stack)==0:
                break
            level,number,lis,pieces,ic= stack.pop()
    solutionlists[(pieces,number)]=res
    return res
#t1=datetime.datetime.now()
#genlist(140,1)
#t2=datetime.datetime.now()
#print str((t2-t1).seconds)
#t3=datetime.datetime.now()
#genlist1(140,1)
#t4=datetime.datetime.now()
#print str((t4-t3).seconds)