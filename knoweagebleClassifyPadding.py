# coding=utf-8
# -*- coding: utf-8 -*-
import codecs
import numpy as np
from codecs import decode
import string
import theano
class CorpusReader:
    def __init__(self, minDocSentenceNum, minSentenceWordNum, dataset="data/split", labelset="data/traindataset2zgb"):
        self.minDocSentenceNum = minDocSentenceNum
        self.minSentenceWordNum = minSentenceWordNum
        # Load labels
        self.labels = loadLabels(labelset)
        print "labels: ", len(self.labels)
          
        # Load documents
        self.docs = loadDocuments(dataset, self.labels.keys(), "GBK")
        print "document: ", len(self.docs)
        
        # Load stop words
        self.stopwords = loadStopwords("data/stopwords", "GBK")
        print "stop words: ", len(self.stopwords)
        
        # Load w2v model data from file
        self.w2vDict = loadW2vModel("data/w2vFlat")
        print "w2v model contains: ", len(self.w2vDict)
        
    labels = None
    docs = None
    stopwords = None
    w2vDict = None
    # print "(570, 301)"
    minDocSentenceNum = 0
    minSentenceWordNum = 0
    __maxDocSentenceNum = 398
    __maxSentenceWordNum = 69
    __wordDim = 200
    __zeroWordVector = [0] * __wordDim
    __zeroSentenceMatrix = [__zeroWordVector] * __maxSentenceWordNum
    
    def getMaxDocSentenceNum(self):
        return self.__maxDocSentenceNum
    
    def getMaxSentenceWordNum(self):
        return self.__maxSentenceWordNum
    
    def getDocNum(self):
        return len(self.labels)
    
    def getDim(self):
        return self.__wordDim
    
    def __sentence2Matrix(self, wordList):
#         f = codecs.open("test.txt", "a+", "utf-8")
#         for str in wordList:
#             f.write(str)
#         f.write(u"\n")
#         f.close()
        sentenceMatrix = map(lambda word: self.w2vDict[word] if (word in self.w2vDict) else None, wordList)
        sentenceMatrix = filter(lambda item: not item is None, sentenceMatrix)
        
        sentenceWordNum = len(sentenceMatrix)
        if(sentenceWordNum < self.minSentenceWordNum):
            return None
        
        if(self.__maxSentenceWordNum > sentenceWordNum):
            sentenceMatrix += [self.__zeroWordVector] * (self.__maxSentenceWordNum - sentenceWordNum)
        elif (self.__maxSentenceWordNum < sentenceWordNum):
            sentenceMatrix = sentenceMatrix[:self.__maxSentenceWordNum]
            sentenceWordNum = self.__maxSentenceWordNum
        return (sentenceMatrix, sentenceWordNum)
    
    def __doc2Matrix(self, e):
        (docIdStr, label) = e
        wordList = self.docs[docIdStr]
        mapL = lambda e: e[1] if (u"\u3000" in e[0] or u"。" in e[0] or u"，" in e[0] or u"." in e[0] or u"," in e[0]) else None
        t = map(mapL, zip(wordList, range(len(wordList))))
        t = filter(None, t)
        t = [0] + t + [len(t)]
        m = map(lambda i:  self.__sentence2Matrix(wordList[i[0] + 1:i[1]]) if(i[1] - i[0] > 1) else None , zip(t[:-1], t[1:]))
        m = filter(lambda item: not item is None, m)
        if(len(m) == 0):
            return None
        docMatrix, sentenceWordNum = zip(*m)
        docMatrix = list(docMatrix)
        sentenceWordNum = list(sentenceWordNum)
        
        docSentenceNum = len(docMatrix)
        if(docSentenceNum < self.minDocSentenceNum):
            return None
        
        if(self.__maxDocSentenceNum > docSentenceNum):
            docMatrix += [self.__zeroSentenceMatrix] * (self.__maxDocSentenceNum - docSentenceNum)
            sentenceWordNum += [0] * (self.__maxDocSentenceNum - docSentenceNum)
        elif (self.__maxDocSentenceNum < docSentenceNum):
            docMatrix = docMatrix[:self.__maxDocSentenceNum]
            sentenceWordNum = sentenceWordNum[:self.__maxDocSentenceNum]
            docSentenceNum = self.__maxDocSentenceNum
        
        return (docMatrix, docSentenceNum, sentenceWordNum, label)
    
    def __getDataMatrix(self, scope):
        scope[1] = np.min([scope[1], len(self.labels)])
        if(scope[0] < 0 or scope[0] >= scope[1]):
            return None
        batch = self.labels.items()[scope[0]:scope[1]]
        docInfo = map(self.__doc2Matrix, batch)
        
        if(len(docInfo) == 0):
            print "Lost doc: ", self.labels.items()[scope[0]:scope[1]]
            return None
        
        docInfo = filter(None, docInfo)
        if(len(docInfo) == 0):
            return None
        
        docMatrixes, docSentenceNums, sentenceWordNums, labels = zip(*docInfo)
        return (docMatrixes, docSentenceNums, sentenceWordNums, labels)
    
    def __findBoarder(self, docSentenceCount, sentenceWordCount):
        maxDocSentenceNum = np.max(docSentenceCount)
        maxSentenceWordNum = np.max(np.max(sentenceWordCount))
        return maxDocSentenceNum, maxSentenceWordNum
    
    # Only positive scope numbers are legal.
    def getCorpus(self, scope):
        # print "(398, 69)"
        # docMatrixes, docSentenceNums, sentenceWordNums, labels = self.__getDataMatrix(scope)
        corpusInfo = self.__getDataMatrix(scope)
#         print len(corpusInfo[0])
#         print self.__findBoarder(corpusInfo[1],corpusInfo[2])
        return corpusInfo
    
def loadDocuments(filename, scope, charset="utf-8"):
    f = open(filename, "r")
    docList = dict()
    for line0 in f:
        try:
            line = decode(line0, charset, 'ignore')
        except:
            continue
        tokens = line.split("\t")
        idStr = tokens[0]
        if(not  idStr in scope):
            continue
        title = tokens[1]
        content = tokens[2]
        docList[idStr] = getWords(title) + getWords(content)
    return docList

def getWords(wordsStr):
    return map(lambda word:   word[:word.index(":")], filter(lambda word: len(word) > 1 and ":" in word, wordsStr.split(" ")))

def loadLabels(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset, "ignore")
    labels = dict()
    for line in f:
        if(not "\t" in line):
            continue
        (k, v) = line.split("\t")
        labels[k] = string.atof(v)
    return labels

def loadStopwords(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset, "ignore")
    d = set()
    for line in f :
        d.add(line)
    return d

def loadW2vModel(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset)
    d = dict()
    for line in f :
        data = line.split(" ")
        word = data[0]
        vec = [string.atof(s) for s in data[1:]]
        d[word] = np.array(vec, dtype=theano.config.floatX)
    return d
