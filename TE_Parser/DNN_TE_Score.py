import sys
import warnings
from utils import getWordmap
from params import params
from utils import getTweetParserData
from utils import getUnlabeledData
from utils import getTagger

import random
import numpy as np
from DNN_TE_Model  import aeparser_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def auencoder( ):
	eta1 = 0.05
	eta2 = 0.005
	window1 =0
	window2 = 1
	numbatch =5
	encodesize = 256




	params.batchsize = 40
	params.hiddensize = 512
	params.layersize = 1024
	params.constraints = False
	params.embedsize = 100
	params.eta1 = eta1
	params.eta2 = eta2
	
	params.save= False
	params.encodesize = encodesize
	params.updatewords = False
	contextsize = window2

	if window2 == 2:
                ReconstructionWeight = [1,1,2,1,1]
        if window2 == 1:
                ReconstructionWeight = [1,2,1]
        if window2 == 3:
                ReconstructionWeight = [1,1,1,2,1,1,1]
	if window2 == 4:
                ReconstructionWeight = [1,1,1,1,2,1,1,1,1]
	(words, We) = getWordmap('embeddings/wordvects.tw100w5-m40-it2')
	words.update({'<s>':0})
	
	a = np.random.rand(len(We[0]))
	root0 = np.zeros(len(We[0]))
	newWe = []
	newWe.append(a.tolist())
	newWe = newWe + We

        newWe.append(root0.tolist())
	We = np.matrix(newWe)
	params.vocabsize = We.shape[0]

	tagger = getTagger('data/tagger')
	


	devdata = getTweetParserData('test', words, tagger)	
       
        tm = aeparser_model(We, params, contextsize, ReconstructionWeight)
        tm.parser(We, devdata, params, window1, contextsize, params.layersize, words)

	
if __name__ == "__main__":
       	auencoder()
