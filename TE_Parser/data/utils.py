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
        if(array[i] in words):
            result.append(words[array[i]])
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
    X = []
    Y = []
    Tag = []
    tag = []
    TokenS = []
    tokens = []
    x = []
    y = []
    unknown = []
    unindex = []
    i =0
    for line in lines:
        i = i+1
        line = line[:-1]
        if len(line)>0:
                line = line.split('\t')
                a = line[1].lower()
                #index = lookupWordsIDX(words, a)
                #x.append(index)
		x.append(a)
                if a not in words:
                        unknown.append(a)
                        unindex.append(i)
                yindex = int(line[6])
		tagis = line[4]
                y.append(yindex)
		tag.append(tagset[tagis])
		tokens.append(int(line[-1]))
        else:
                X.append(x)
                Y.append(y)
		Tag.append(tag)
		TokenS.append(tokens)
                x = []
                y = []
		tag = []
		tokens = []

    return X, Y, Tag, TokenS, unknown, unindex




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


