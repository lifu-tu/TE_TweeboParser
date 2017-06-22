import theano
import numpy as np
from theano import tensor as T
from theano import config
import pdb
import random as random
import string
#from evaluate import evaluate_all
import time
from utils import lookupwordID
from lasagne_embedding_layer_2 import lasagne_embedding_layer_2
#from LSTMLayerNoOutput import LSTMLayerNoOutput
from collections import OrderedDict
import lasagne
import sys
import cPickle
import pickle

def Acc(pred, y, tokenS):
        testy = [s0 for s in y for s0 in s]
	goldST = [(s0>=0) for s in y for s0 in s]
        tmp = [ sum(s) for s in tokenS]
        lengths = [len(s) for s in y]
        prednew = []
        p = 0
        index = 0
        for idx, s in enumerate(tokenS):
                for e in s:
                        if e ==0:
                                prednew.append(-3)

                        elif tmp[idx] == 1:
                                prednew.append(0)

                        elif pred[p] == lengths[idx]:
                                prednew.append(0)
                                p = p + 1
                        else :
                                #prednew[index] = pred[p] +1
                                prednew.append(pred[p]+1)
                                p = p + 1
        finalpred = []
        for length in lengths:
                a = []
                for i in range(length):
                        a.append(prednew[index])
                        index = index + 1
                finalpred.append(a)

        count = 0
        for idx , s in enumerate(testy):
                if (s == prednew[idx]) and (goldST[idx]==1):
                        count = count + 1
        acc = 1.0*count/sum(goldST)
        recal = 1.0 * count/ sum(tmp)
        f1 = 2.0*acc * recal / (acc + recal)
        return finalpred, f1


#random.seed(1)
#np.random.seed(1)

def Score(score, tokenS):
        tmp = [ sum(s) for s in tokenS]
        lengths = [len(s) for s in tokenS]
        prednew = []
        p = 0
        index = 0
        for idx, s in enumerate(tokenS):

		f = open('../working_dir/test_score1/Token_'+ str(idx), 'w')
                for idx1, e in enumerate(s):

                        if (e ==1) and (tmp[idx] == 1):
                                f.write('0\t'+ str(idx1+1)+ '\t1\n')

                        elif (e==1) and (tmp[idx] != 1):
                                f.write( '0\t'+ str(idx1+1)+ '\t'+ str(score[p,lengths[idx]])+'\n')
                                for i in range(lengths[idx]):
                                        if (s[i]==1) and (i!=idx1):
                                        #if (i!=idx1):
                                                f.write( str(i+1) +'\t'+ str(idx1+1)+ '\t'+ str(score[p,i])+'\n')
                                p = p + 1


                f.close()



maxlen = 40

def checkIfQuarter(idx,n):
    #print idx, n
    if idx==round(n/4.) or idx==round(n/2.) or idx==round(3*n/4.):
        return True
    return False

class aeparser_model(object):

    #takes list of seqs, puts them in a matrix
    #returns matrix of seqs and mask	
    def prepare_data(self, list_of_seqs,labels, contextsize):
        lengths = [len(s) for s in list_of_seqs]
	sumlength = sum(lengths)
        n_samples = len(list_of_seqs)
        x = np.zeros((sumlength, 2*contextsize+1)).astype('int32')
        y = np.zeros(sumlength).astype('int32')
	index = 0
        for i in range(n_samples):
	    new_seq = [0]*contextsize + list_of_seqs[i] + [1]*contextsize
            for j in range(lengths[i]):
            	x[index, :] = new_seq[j:j+2*contextsize+1]
            	y[index] = labels[i][j]
		index = index + 1
	#print len(labels)
        return x, y, n_samples

    def prepare_aedata(self, list_of_seqs, contextsize):
	lengths = [len(s) for s in list_of_seqs]
        sumlength = sum(lengths)
        n_samples = len(list_of_seqs)
        x = np.zeros((sumlength, 2*contextsize+1)).astype('int32')
        index = 0
        for i in range(n_samples):
            new_seq = [0]*contextsize + list_of_seqs[i] + [1]*contextsize
            for j in range(lengths[i]):
                x[index, :] = new_seq[j:j+2*contextsize+1]
                index = index + 1
        #print len(labels)
        return x, n_samples
    
    def prepare_partiondata(self, list_of_seqs, tagger, tokenS, contextsize, vocabsize, words):
	
	       
        lengths = [len(s) for s in list_of_seqs]
        sumlength = 0
	tmp = [ sum(s) for s in tokenS]
	for index, s in enumerate(tmp):
                if s > 1:
                        sumlength = sumlength + s
                #else:
                #       print list_of_seqs[index]
        n_samples = len(list_of_seqs)
        maxlen0 = np.max(lengths)
        maxindex = np.argmax(lengths)
        if maxlen0 > maxlen:
                print list_of_seqs[maxindex]
                print 'too long tweet: ', maxlen0
                sys.exit()

        window = 2*(2*contextsize+1)
        x = np.zeros((sumlength, maxlen+1, window)).astype('int32')
        D = np.zeros((sumlength, maxlen+1,80)).astype('int32')
        x_mask = np.zeros((sumlength,  maxlen+1)).astype('float32')
       
        idx = 0
        for i in range(n_samples):
	    if tmp[i] <=1:
                continue
           
            new_tagger = tagger[i]
            list_of_seqs1 = lookupwordID(words,list_of_seqs[i])
            #new_tag = tagger[i]
            new_seq = [0]*contextsize + list_of_seqs1 + [1]*contextsize
            for j in range(lengths[i]):
		if tokenS[i][j] == 0:
                        continue
                x0 = new_seq[j:j+2*contextsize+1]
                x00 = np.zeros(36).astype('int32')
                word_j = list_of_seqs[i][j]
                #jt = int(1.0*j/lengths[i]*5)
                #x00[jt]=1
                punc_flag = 1
                
		a = 0
                for s in word_j:
                        if s in string.punctuation:
                                a = a +1
                if a == len(word_j):
                        punc_flag = 0
                if word_j =='<@MENTION>':
                        x00[0]=1
                elif (word_j[0] =='#') and (len(word_j)!= 1):
                        x00[1]=1
                elif word_j =='rt':
                        x00[2]=1
                elif 'URL' in word_j:
                        x00[3]=1
                elif (word_j.replace('.','',1).isdigit())  or (word_j.replace(',','',1).isdigit()):
                        x00[4]=1
                # check whether it is punc
                elif '$' in word_j:
                        x00[5]=1
                elif word_j ==':':
                        x00[7]=1
                elif word_j =='...':
                        x00[8]=1
                elif (len(word_j) == 1) and (word_j[0] in string.punctuation):
                        x00[9]=1
                elif punc_flag == 0:
                        x00[6]= 1

                x00[10] = 1.0 *j / lengths[i]
                x00[11 + new_tagger[j]] =  1

		for k in range(lengths[i]+2):
                        D[idx,k,:36] = x00
                        if (k!=j) and (k< lengths[i]):

                                word_j = list_of_seqs[i][k]
                                #kt = int(1.0*k/lengths[i]*5)
                                #D[idx, k, 11+kt]=1
                                punc_flag = 1
                                a = 0
                                for s in word_j:
                                        if s in string.punctuation:
                                                a = a +1
                                if a == len(word_j):
                                        punc_flag = 0
                                if word_j =='<@MENTION>':
                                        D[idx, k,36]=1
                                elif (word_j[0] =='#') and (len(word_j)!= 1):
                                        D[idx,k, 37]=1
                                elif word_j =='rt':
                                        D[idx,k,38]=1
                                elif 'URL' in word_j:
                                        D[idx,k,39]=1
                                elif (word_j.replace('.','',1).isdigit())  or (word_j.replace(',','',1).isdigit()):
                                        D[idx,k, 40]=1
                # check whether it is punc
                                elif '$' in word_j:
                                        D[idx,k,41]=1
                                elif word_j ==':':
                                        D[idx,k,43]=1
                                elif word_j =='...':
                                        D[idx,k,44]=1
                                elif (len(word_j) == 1) and (word_j[0] in string.punctuation):
                                        D[idx,k,45]=1
                                elif punc_flag == 0:
                                        D[idx,k,42]= 1
				
				x[idx,k,:] = x0 + new_seq[k:k+2*contextsize+1]
                                #D[idx, k, 14]= abs(k-j)
                                D[idx, k, 46] =  1.0 * k / lengths[i]
                                D[idx, k, 47 + new_tagger[k]] = 1
                                if abs(k-j)==1:
                                        D[idx,k,72] =1
                                elif abs(k-j)==2:
                                        D[idx,k,73] =1
                                elif (abs(k-j)>2) and (abs(k-j)<6):
                                        D[idx,k,74] =1
                                elif (abs(k-j)>5) and (abs(k-j)<11):
                                        D[idx,k,75] =1
                                elif (abs(k-j)>10):
                                        D[idx,k,76] =1

				if tokenS[i][k] ==1:
                                        x_mask[idx,k]=1.


                        if k == lengths[i]:
                                x[idx,k,:]= x0 + [vocabsize-1]*(2*contextsize+1)
                                D[idx,k,77]=1
				x_mask[idx,k]=1.
			elif k < j:
                                D[idx,k,78]= 1
                        elif k > j:
                                D[idx,k,79]= 1

               
                idx = idx +1

        return x, x_mask, D



    def saveParams(self, para, fname):
        f = file(fname, 'wb')
        cPickle.dump(para, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)


    def __init__(self, We_initial, params, contextsize, ReconstructionWeight):

        self.layersize = params.hiddensize
        self.memsize = params.embedsize
        self.eta = params.eta1
        ReconstructionWeight = np.array(ReconstructionWeight)
        self.we = theano.shared(We_initial)
        
	g1 = T.imatrix()
        l_in = lasagne.layers.InputLayer((None, 2 * contextsize + 1))
        #l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.we.get_value().shape[0], output_size=self.we.get_value().shape[1], W=self.we)
        l_emb = lasagne_embedding_layer_2(l_in, self.we.eval().shape[1], self.we)

        l_0 = lasagne.layers.ReshapeLayer(l_emb, (-1, (2 * contextsize + 1)*self.memsize))
        #print l_0.output_shape
        #l_enc1 = lasagne.layers.DenseLayer(l_0, self.layersize)
        l_enc2 = lasagne.layers.DenseLayer(l_0, self.layersize)
        l_newemb = lasagne.layers.DenseLayer(l_enc2, params.encodesize)
        #l_dnc1 = lasagne.layers.DenseLayer(l_newemb, self.layersize)
        l_dnc2 = lasagne.layers.DenseLayer(l_newemb, self.layersize)
        l_out = lasagne.layers.DenseLayer(l_dnc2,(2 * contextsize + 1)*self.memsize, nonlinearity=lasagne.nonlinearities.linear)
        l_out0 = lasagne.layers.ReshapeLayer(l_out, (-1, 2*contextsize + 1, self.memsize))


        self.encoder = lasagne.layers.get_all_params(l_newemb, trainable=True)
        #self.encoder.pop(0)
        #l_out = lasagne.layers.DenseLayer(l_2, 25, nonlinearity=lasagne.nonlinearities.softmax)
        n_sentenses = T.scalar()
        #objective function
        #output = lasagne.layers.get_output(l_out)

        output0 = lasagne.layers.get_output(l_out0, {l_in:g1})
        oldemb = lasagne.layers.get_output(l_emb, {l_in:g1})
        oldemb0 = lasagne.layers.get_output(l_0, {l_in:g1})
        newemb = lasagne.layers.get_output(l_newemb, {l_in:g1})

        #pred = T.argmax(output, axis =1)
        #cost = cost + self.LW * lasagne.regularization.l2(self.we - We0)
        cost0 = lasagne.objectives.squared_error(output0, oldemb)
        cost1 = T.sum(cost0, axis = 2)
        #cost2 = T.sum(cost1, axis =0)
        cost2 = T.mean(cost1, axis =0)
        cost = T.sum(cost2 * ReconstructionWeight)
        #cost = 1.0 * cost / n_sentenses
        self.encode = lasagne.layers.get_all_params(l_newemb, trainable=True)

        network_params = lasagne.layers.get_all_params(l_out, trainable=True)
        self.all_params = network_params
        #network_params.pop(0)

	self.cost_function = theano.function([g1, n_sentenses], cost, on_unused_input='warn')
        #updates

        updates = lasagne.updates.sgd(cost, network_params, self.eta)
        updates = lasagne.updates.apply_momentum(updates, network_params, momentum=0.9)
        self.train_function = theano.function([g1, n_sentenses], cost, updates=updates, on_unused_input='warn')
       
    
    def parser(self, We_initial, devdata, params, window1, contextsize, layersize, words):

        initial_We = theano.shared(We_initial).astype(theano.config.floatX)

	devx,  devTag , devTokenS = devdata
	#devx0, devx0_mask, devx0_mask1, devD0, devy0 = self.prepare_partiondata(devx, devy, devTag, contextsize, params.vocabsize, words)
	devx00, devx00_mask, devD00 = self.prepare_partiondata(devx, devTag, devTokenS, window1, params.vocabsize, words)
        devx000 = devx00[:,:, :(2*window1+1)]
        devx001 = devx00[:,:, (2*window1+1):]
	devx01, _, _ = self.prepare_partiondata(devx, devTag, devTokenS, contextsize, params.vocabsize, words)
	devx010 = devx01[:,:, :(2*contextsize+1)]
        devx011 = devx01[:,:, (2*contextsize+1):]
	
	

	contextsize1 = window1
	g0 = T.itensor3()
        g1 = T.itensor3()
	g20 = T.itensor3()
        g21 = T.itensor3()
        masksum = T.dmatrix()
      
        d1sum = T.itensor3()
      

       
        l_in0 = lasagne.layers.InputLayer((None, maxlen +2, 2 * contextsize1 + 1))
        #l_emb = lasagne.layers.EmbeddingLayer(l_in0, input_size=self.we.get_value().shape[0], output_size=self.we.get_value().shape[1], W=self.we)
        l_emb = lasagne_embedding_layer_2(l_in0, self.we.eval().shape[1], self.we)
	#l_0 = lasagne.layers.ReshapeLayer(l_emb, (-1, (2 * contextsize + 1)*self.memsize))
        l_0 = lasagne.layers.FlattenLayer(l_emb,3)
        oldv0 = lasagne.layers.get_output(l_0, {l_in0:g0})
        oldv1 = lasagne.layers.get_output(l_0, {l_in0:g1})


	g01 = g20.reshape((-1, 2 * contextsize + 1))
	g11 = g21.reshape((-1, 2 * contextsize + 1))
	l_in1 = lasagne.layers.InputLayer((None,  2 * contextsize + 1))
        #l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, input_size=self.we.get_value().shape[0], output_size=self.we.get_value().shape[1], W=self.we)
       	l_emb1 = lasagne_embedding_layer_2(l_in1, self.we.eval().shape[1], self.we)
        l_01 = lasagne.layers.ReshapeLayer(l_emb1, (-1, (2 * contextsize + 1)*self.memsize))
	#l_01 = lasagne.layers.FlattenLayer(l_emb1,3)
        l_enc2 = lasagne.layers.DenseLayer(l_01, self.layersize)
        l_newemb1 = lasagne.layers.DenseLayer(l_enc2, params.encodesize)
	
	f = open('models/hiddensize_512dnn_parser_fixembeding_etype_0_evaType_1_window2_1_LearningRate_0.05_0.005.pickle','r')
        PARA = pickle.load(f)
        #PARA = [p.get_value() for p in para]
        f.close()

        encoder = lasagne.layers.get_all_params(l_newemb1, trainable=True)
	#encoder.pop(0)
	for idx, p in enumerate(encoder):
		p.set_value(np.float32(PARA[idx]))
       
	
	newv00 = lasagne.layers.get_output(l_newemb1, {l_in1:g01})
	newv10 = lasagne.layers.get_output(l_newemb1, {l_in1:g11})
	newv0 = newv00.reshape((-1, maxlen + 1 , params.encodesize))
	newv1 = newv10.reshape((-1, maxlen + 1, params.encodesize))

        newxsum00 = T.concatenate((newv0, newv1, d1sum), axis = 2)
	#print newxsum00.shape

	tokensize = 2*params.encodesize + 80
	newxsum0 = newxsum00.reshape((-1, tokensize))
	

	l_newin = lasagne.layers.InputLayer((None, tokensize))
	l_1 = lasagne.layers.DenseLayer(l_newin, layersize)
        l_2 = lasagne.layers.DenseLayer(l_1, layersize)
	l_out = lasagne.layers.DenseLayer(l_2, 1, nonlinearity=lasagne.nonlinearities.linear)



	score1sum = lasagne.layers.get_output(l_out, {l_newin:newxsum0})
        score1sum = score1sum.reshape((-1,maxlen+1))

	#score1sum = 0.5 * score1sum

        score1sum0 = T.exp(score1sum)
        score1sum0 = score1sum0 * masksum
      
        pred = T.argmax(score1sum0, axis =1)
        
	sum_score = T.sum(score1sum0, axis=1)
        normal_score = score1sum0/sum_score[:, None]

	
	

	#cla_acc_function = theano.function([g0, g1, g20, g21, masksum, d1sum, y0], acc, on_unused_input='warn')	
	c_params0 = lasagne.layers.get_all_params(l_out, trainable=True)
	c_params1 = lasagne.layers.get_all_params(l_newemb1, trainable=True)
	c_params = c_params0
	f = open('models/hiddensize_512dnn_parser_fixembeding_etype_0_evaType_1_window2_1_LearningRate_0.05_0.005__pretestParser.pickle','r')
        PARA = pickle.load(f)
        f.close()

        for idx, p in enumerate(c_params):
                p.set_value(PARA[idx])

	cla_score_function = theano.function([g0, g1, g20, g21, masksum, d1sum], normal_score, on_unused_input='warn')	

	parser_score = cla_score_function(devx000, devx001, devx010, devx011, devx00_mask, devD00)
	Score(parser_score,  devTokenS)
	print 'Seen %d samples' % len(devTokenS)	
