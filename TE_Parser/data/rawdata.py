
import sys
def RawTweetParserData():
    data = open(sys.argv[1],'r')
    lines = data.readlines()
    data.close()

    out = open(sys.argv[2],'w')
   
    x = ''

    i =0
    for line in lines:
        i = i+1
        line = line[:-1]
        if len(line)>0:
                line = line.split('\t')
                a = line[1]
                x=x + a +' '
        else:
		x = x.strip()
                out.write(x+'\n')
                x = ''
    out.close() 

if __name__=="__main__":
    RawTweetParserData()
