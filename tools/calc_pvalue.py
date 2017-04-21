import re
import numpy as np
# From KDD 2014
# pdvalue(pvalue of some attribute)-> pvalue  
def calc(inputfile,outputfile,duration=1):
    def I(compare):
        if compare:
            return 1
        else:
            return 0
    with open(inputfile,'r') as fr, open(outputfile,'w+') as fw:
        for line in fr.readlines():
            pvalues=[]
            p1=re.compile(r"\(.*?\)")
            p2=re.compile("[\d\.]+")
            raw_data= [[float(ele) for ele in re.findall(p2,aline)]\
                                    for aline in re.findall(p1,line)]
            for day,a in enumerate(raw_data):
                if day<duration:
                    continue
                D=len(raw_data[day])
                # a: one day v's observations, calc out its pvalue
                days_set= np.arange(day-duration,day+1)
                min_pdvalues={}
                for dayq in days_set:
                    min_pdvalues[dayq]=min([   np.average([ I(raw_data[ob_day][d]>= raw_data[dayq][d])   for ob_day in days_set[:-1]  ])        for d in range(D)])
                pvalues.append(np.average([ I(min_pdvalues[ob_day]<=min_pdvalues[day])for ob_day in days_set[:-1] ]) )

            fw.write(' '.join([ str(value) for value in pvalues])+'\n')
if __name__=="__main__":
    #calc("1","2",3)

    
