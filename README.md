repl4NLP2017

Code for token embedding parser from "Learning to Embed Words in Context for  Syntactic Tasks"

The code is written in python and requires numpy, theano, and the lasagne libraries. And Some of 
the code is from the following project. (https://github.com/ikekonglp/TweeboParser)

A Dependency Parser for Tweets
Lingpeng Kong, Nathan Schneider, Swabha Swayamdipta, Archna Bhatia, Chris Dyer, and Noah A. Smith. In Proceedings of EMNLP 2014.



If you use the code for your work please cite:

@inproceedings{tu-17,
  title={Learning to Embed Words in Context for Syntactic Tasks},
  author={Lifu Tu and Kevin Gimpel and Karen Livescu},
  booktitle={Proc. of RepL4NLP},
  year={2017}
}



@inproceedings{tu-17-long,
  title={Learning to Embed Words in Context for Syntactic Tasks},
  author={Lifu Tu and Kevin Gimpel and Karen Livescu},
  booktitle={Proceedings of the 2nd Workshop on Representation Learning for NLP},
  year={2017},
  publisher = {Association for Computational Linguistics}
}

@InProceedings{kong2014dependency,
  title={A dependency parser for tweets},
  author={Kong, Lingpeng and Schneider, Nathan and Swayamdipta, Swabha and Bhatia, Archna and Dyer, Chris and Smith, Noah A.},
  booktitle={Proc. of EMNLP},
  year={2014}
}



Please refer to the paper for more information.

An example of a dependency parse of a tweet is:

![Alt text](http://www.cs.cmu.edu/~ark/TweetNLP/deptree.jpg)

Corresponding CoNLL format representation of the dependency tree above:

```
1       OMG     _       !       !       _       0       _
2       I       _       O       O       _       6       _
3       ♥       _       V       V       _       6       CONJ
4       the     _       D       D       _       5       _
5       Biebs   _       N       N       _       3       _
6       &       _       &       &       _       0       _
7       want    _       V       V       _       6       CONJ
8       to      _       P       P       _       7       _
9       have    _       V       V       _       8       _
10      his     _       D       D       _       11      _
11      babies  _       N       N       _       9       _
12      !       _       ,       ,       _       -1      _
13      —>      _       G       G       _       -1       _
14      LA      _       ^       ^       _       15      MWE
15      Times   _       ^       ^       _       0       _
16      :       _       ,       ,       _       -1      _
17      Teen    _       ^       ^       _       19      _
18      Pop     _       ^       ^       _       19      _
19      Star    _       ^       ^       _       20      _
20      Heartthrob      _       ^       ^       _       21      _
21      is      _       V       V       _       0       _
22      All     _       X       X       _       24      MWE
23      the     _       D       D       _       24      MWE
24      Rage    _       N       N       _       21      _
25      on      _       P       P       _       21      _
26      Social  _       ^       ^       _       27      _
27      Media   _       ^       ^       _       25      _
28      …       _       ,       ,       _       -1      _
29      #belieber       _       #       #       _       -1      _
```
(HEAD = -1 means the word is not included in the tree)



##Compiling

```
Note: You will need the latest GCC, cmake, Java, Python, theano, Lasagne to run the code. If you find problems, please send a email to lifu@ttic.edu
```

run the following command

```
> ./install.sh
```

This will install the parser and all its dependencies. Also, it will download some pretrained models for you. Some of them are from http://www.cs.cmu.edu/~ark/TweetNLP/pretrained_models.tar.gz

##Example of usage

To run the Parser on raw text input with one sentence per line (e.g. on the
sample_input.txt):

```
> ./run.sh sample_input.txt
```

The run.sh file contains the steps we run the whole TweeboParser pipeline, including
twokenization, POS tagging, appending brown clustering features and PTB features etc.

The output file will be "sample_input.txt.predict" in the same directory as
"sample_input.txt".

which contains the CoNLL format (http://ilk.uvt.nl/conll/#dataformat) output of the
parse tree. (HEAD < 0 means the word is not included in the tree)

##Directory Structure
```

TE_Parser               ---     The token embedding model for the parser, which provide the arc score 
						under our model from our head predictors
ark-tweet-nlp		----	The Twitter POS Tagger (http://www.ark.cs.cmu.edu/TweetNLP/)
pretrained_models	----	Tagging, token selection, brown clusters obtained from
				Owoputi et al. (2012) and pre-trained parsing models for PTB
							and Tweets.
scripts			----	Supporting scripts.
TBParser		----	TweeboParser, which is based on TurboParser version 2.1.0.
							The source code of TweeboParser can be found at TBParser/src
token_selection		----	The token selection tool implemented in Python.
Tweebank		----	Tweebank data release.
working_dir		----	The working space for the parser. The temp files generated by
				TweeboParser when parsing a sentence are putted here. (So don't
							remove or rename it.)
run.sh			----	The bash script which runs the parser on raw inputs
install.sh		----	The bash script which installs everything
```
