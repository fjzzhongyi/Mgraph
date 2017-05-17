import re,sys
import os,shutil
def add(srcdir,func=None):
    srcdir=srcdir.rstrip(os.sep)
    sdspark=None
    for syspath in sys.path:
        p=os.path.join(syspath,"sdspark")
        if os.path.exists(p) and os.path.isdir(p):
            sdspark=p
            break
    if sdspark==None:
        print "plz ensure MGraph is within your system paths list"
        return False

    despath=os.path.join(sdspark,os.path.basename(srcdir))
    #copy
    if not os.path.exists(srcdir):
        print "source dir not exists"
        return False
    elif not os.path.isdir(srcdir):
        print "source path is not a dir"
        return False
    elif not os.path.isabs(srcdir):
        print "absolute path is needed"
        return False
    
    if os.path.exists(os.path.join(despath)):
        print "this algo has exist. you need rename your dir"
        return False

    shutil.copytree(srcdir,despath)
    
    files= filter(lambda x :True if os.path.splitext(x)[1]==".py" else False,[os.path.join(despath,f) for f in  os.listdir(despath)])
    modules=[ os.path.splitext(os.path.basename(f))[0] for f in files]
    
    with open(os.path.join(despath,"__init__.py"),'w+') as f:
        f.write("__all__=[\"")
        f.write("\", \"".join(modules))
        f.write("\"]")
        
    with open(os.path.join(sdspark,"__init__.py"),"w+") as f:
        files1 = filter(lambda x: True if os.path.isdir(x) or os.path.splitext(x)[1]==".py" else False, [os.path.join(sdspark,fi) for fi in os.listdir(sdspark)])
        modules1=[ os.path.splitext(os.path.basename(fi))[0] for fi in files1]
        f.write("__all__=[\"")
        f.write("\", \"".join(modules1))
        f.write("\"]")
    prefix="sdspark."+os.path.basename(srcdir)+"."
    
    for f in files:
        with open(f,'r')as fr:
            lines=fr.readlines()
        with open(f,'w+') as fw:
            for line in lines:
                #n =filter(lambda x: True if len(x)>0 else False,re.split(r" |\t",line))
                for m in modules:
                    line=line.replace(m,prefix+m)
                fw.write(line)
    print "done"
    return True
                
if __name__=="__main__":
    add("/home/hongyi/Desktop/testSpark")
