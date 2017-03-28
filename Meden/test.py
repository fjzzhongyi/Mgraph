import commands

if __name__ == "__main__":
    command="java -Xms1024m -Xmx2048m -jar meden.jar -run traffic-small.txt 0 10 p 0.1"
    line = str(commands.getoutput(command))
    print line
