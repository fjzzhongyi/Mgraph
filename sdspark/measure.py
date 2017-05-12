import time
import logging    
import psutil  
import os  
import thread
import psutil
class logger:
    def __init__(self,outdir): 
        #base= os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        base=outdir
        self.file = open(os.path.join(base,'cpu_memo_log'),'w+')
        t1=thread.start_new_thread(self.log,())
    def get(self):
        return self.cpu_per,self.memo_per

    def log(self):
        while True:         
            # get pid  
            p1=psutil.Process(os.getpid())
            # cpu rss vms
            self.file.write(str(time.time())+' '+str(p1.cpu_percent(interval=0.05))+' '+str(p1.memory_info()[0])+' '+str(p1.memory_info()[1])+'\n')
            self.file.flush()
            #print cpu,memo
            time.sleep(0.5)
