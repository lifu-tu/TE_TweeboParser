from scipy.io import loadmat
import numpy as np
import math
from random import shuffle
from random import choice
from random import randint
from theano import tensor as T

def lookupWordsIDX(words,w):
    if w in words:
        return words[w]
    else:
        #print 'find UUUNKKK words',w
        return words['uuunkkk']

def lookupTaggerIDX(tagger,w):
    w = w.lower()
    if w in tagger:
        return tagger[w]
    else:
        #print 'find UUUNKKK words',w
        return tagger['*']

def lookup_with_unk(We,words,w):
    if w in words:
        return We[words[w],:],False
    else:
        #print 'find Unknown Words in WordSim Task',w
        return We[words['uuunkkk'],:],True

def lookupwordID(words,array):
    #w = w.strip()
    result = []
    for i in range(len(array)):
        if(array[i].lower() in words):
            result.append(words[array[i].lower()])
        else:
            #print "Find Unknown Words ",w
            result.append(words['uuunkkk'])
    return result

def lookupTaggerID(tagger, array):
    #w = w.strip()
    result = []
    for i in range(len(array)):
        if(array[i] in tagger):
            result.append(tagger[array[i]])
        else:
            #print "Find Unknown tagger *
            result.append(tagger['*'])
    return result


def getData(f, words, tagger):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
	    index = i.find('|||')
	    if index == -1:
		print('file error\n')
		return None
	    x = i[:index-1]
	    y = i[index+4:-1]
	    x = x.split(' ')
	    y = y.split(' ')
            #print x
	    #print y
	    x = lookupwordID(words, x)
            y = lookupTaggerID(tagger, y)
	    #print y
            X.append(x)
	    Y.append(y)
   
    return X, Y


def getTweetParserData(f, words, tagset):
    data = open(f,'r')
    lines = data.readlines()
    data.close()
    X = []
    Tag = []
    tag = []
    TokenS = []
    tokens = []
    x = []

    i =0
    for line in lines:
        i = i+1
        line = line[:-1]
        if len(line)>0:
                line = line.split('\t')
		x.append(line[1])
		tagis = line[4]
		tag.append(tagset[tagis])
		tokens.append(int(line[-1]))
        else:
                X.append(x)
		Tag.append(tag)
		TokenS.append(tokens)
                x = []
		tag = []
		tokens = []

    return X, Tag, TokenS



def Cluster(filename):
    data = open(filename,'r')
    lines = data.readlines()
    data.close()
    cluster = {}
    #cluster_n = {}
    #X = 0
    #x = 0
    for index, i in enumerate(lines):
        if(len(i) > 0):
            i = i.strip()
            i = i.split('\t')
            a = i[1].lower()
            if a not in cluster:
                b = i[0] + '0'*(16- len(i[0]))
                #c = np.zeros(16).astype('int32')
                #for ii, bb in enumerate(b):
                #       c[ii] = int(bb)
                cluster[a] = b
    return cluster


def getgram_dic(dicfile):
    dic = {}
    f = open(dicfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i[:-1]
        dic[i] = n
    return dic


def getUnlabeledData(f, words):
    data = open(f,'r')
    lines = data.readlines()
    X = []
    Y = []
    for i in lines:
        if(len(i) > 0):
            x = i[:-1]
            x = x.split(' ')
            x = lookupwordID(words, x)
            #print y
            X.append(x)
    return X

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0].lower()]=n+1
        We.append(v)
    return (words, We)

def getTagger(Taggerfile):
    tag = {}
    f = open(Taggerfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i = i.strip()
        tag[i] = n
    return tag


def getVec(We,words,t):
    t = t.strip()
    array = t.split(' ')
    if array[0] in words:
        vec = We[words[array[0]],:]
    else:
        #print 'find UUUNKKK words',array[0].lower()
        vec = We[words['UUUNKKK'],:]
    for i in range(len(array)-1):
        #print array[i+1]
        if array[i+1] in words:
            vec = vec + We[words[array[i+1]],:]
        else:
            #print 'can not find corresponding vector:',array[i+1].lower()
            vec = vec + We[words['UUUNKKK'],:]
    vec = vec/len(array)
    return vec


def ReluT(x):
    return T.switch(x<0, 0 ,x)

def Relu(x):
    result = np.zeros(x.shape)
    #print x.shape
    for i in xrange(result.shape[0]):
        if(x[i]>0):
            result[i]=x[i]
    return result

def Sigmoid(x):
    result = np.zeros(x.shape)
    for i in xrange(result.shape[0]):
        for j in xrange(result.shape[1]):
            result[i][j] = 1 / (1 + math.exp(-x[i][j]))
    return result


