import time
import logging    
import psutil  
import os  
import thread
import psutil
class logger:
    def __init__(self): 
        base= os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        self.file = open(os.path.join(base,'cpu_memo_log'),'w+')
        t1=thread.start_new_thread(self.log,())
    def get(self):
        return self.cpu_per,self.memo_per

    def log(self):
        while True:         
            # get pid  
            p1=psutil.Process(os.getpid())
            # cpu rss vms
            self.file.write(str(p1.cpu_percent(interval=0.05))+' '+str(p1.memory_info()[0])+' '+str(p1.memory_info()[1])+'\n')
            #print cpu,memo
time.sleep(1)
