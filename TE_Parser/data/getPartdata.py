
import sys
def RawTweetParserData():
    data = open(sys.argv[1],'r')
    lines = data.readlines()
    data.close()

    out1 = open('part_train','w')
    out2 = open('part_test','w')   


    #file1 = out1
    flag =0
    i =0
    for line in lines:
        #i = i+1
        #line = line[:-1]
        if len(line)>1:
		if i < 100:
			out2.write(line)
		else:
			flag =1
			out1.write(line)
    
        else:
		i = i+1
		if flag ==0:
			out2.write(line)
		else:
			out1.write(line)
		#x = x.strip()
                #out.write(x+'\n')
    print i
    out1.close() 
    out2.close()

if __name__=="__main__":
    RawTweetParserData()
