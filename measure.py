import time
import logging    
import psutil  
import os  
import threading
import psutil
class logger:
    def __init__(self): 
        self.cpu_per=0
        self.memo_per=0
        self.times=0
        threads=[]
        t1=threading.Thread(target=self.log)
    def get(self):
        return self.cpu_per,self.memo_per

    def log(self):
        while True:         
            # get pid  
            p1=psutil.Process(os.getpid())
            
            self.cpu_per=cpu=(self.cpu_per*self.times+p1.cpu_percent(None))/float(self.times+1)
            self.memo_per=memo=(self.memo_per*self.times+p1.memory_percent())/float(self.times+1)
            self.times+=1
            print cpu,memo
            time.sleep(1)
