import numpy
import theano
from theano import tensor as T, printing
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class DocEmbeddingNN:
    
    def __init__(self,
                          corpus,
                          docSentenceCount,
                          sentenceWordCount,
                          rng,
                          corpus_shape,
                          maxRandge=10000,
                          sentenceLayerNodesNum=2,
                          sentenceLayerNodesSize=5,
                          docLayerNodesNum=2,
                          docLayerNodesSize=2,
                          datatype=theano.config.floatX):
        self.__sentenceLayerNodesNum = sentenceLayerNodesNum
        self.__sentenceLayerNodesSize = sentenceLayerNodesSize
        self.__docLayerNodesNum = docLayerNodesNum
        self.__docLayerNodesSize = docLayerNodesSize
        self.__WBound = 0.2
        self.__MAXDIM = maxRandge
        self.__datatype = datatype
        self.sentenceW = None
        self.sentenceB = None
        self.docW = None
        self.docB = None
        
        self.shareRandge = T.arange(maxRandge)
        
        # Get sentence layer W
        self.sentenceW = theano.shared(
            numpy.asarray(
                rng.uniform(low=-self.__WBound,
                                        high=self.__WBound,
                                        size=(sentenceLayerNodesNum,
                                                    corpus_shape[3],
                                                    1,
                                                    sentenceLayerNodesSize)),
                dtype=datatype
            ),
            borrow=True
        )
        # Get sentence layer b
        sentenceB0 = numpy.zeros((sentenceLayerNodesNum,), dtype=datatype)
        self.sentenceB = theano.shared(value=sentenceB0, borrow=True)
        
        # Get doc layer W
        self.docW = theano.shared(
            numpy.asarray(
                rng.uniform(low=-self.__WBound,
                                        high=self.__WBound,
                                        size=(docLayerNodesNum,
                                                    sentenceLayerNodesNum,
                                                    docLayerNodesSize,
                                                    1)),
                dtype=datatype
            ),
            borrow=True
        )
        # Get doc layer b
        docB0 = numpy.zeros((docLayerNodesNum,), dtype=datatype)
        self.docB = theano.shared(value=docB0, borrow=True)
        
        corpus = corpus.dimshuffle([0, 3, 1, 2])

        sentenceConv = conv.conv2d(
            input=corpus,
            filters=self.sentenceW,
            image_shape=[None, corpus_shape[3], corpus_shape[1], corpus_shape[2]],
            filter_shape=[sentenceLayerNodesNum, corpus_shape[3], 1, sentenceLayerNodesSize]
        )
        
        sentencePooled_out = downsample.max_pool_2d(
            input=sentenceConv,
            ds=(1, maxRandge)
        )
        
        sentenceOutput = T.tanh(sentencePooled_out + self.sentenceB.dimshuffle('x', 0, 'x', 'x'))
        
        docConv = conv.conv2d(
            input=sentenceOutput,
            filters=self.docW,
            image_shape=[None, self.__sentenceLayerNodesNum, corpus_shape[1], 1],
            filter_shape=[docLayerNodesNum,
                                                    sentenceLayerNodesNum,
                                                    docLayerNodesSize,
                                                    1]
        )

        
        docPooled_out = downsample.max_pool_2d(
            input=docConv,
            ds=(maxRandge, 1)
        )
        
        docOutput = T.tanh(docPooled_out + self.docB.dimshuffle('x', 0, 'x', 'x'))
        
        
        self.output = docOutput.flatten(2)
        
#         p = printing.Print('self.output')
#         self.output = p(self.output)
        self.params = [self.sentenceW, self.sentenceB, self.docW, self.docB]
        self.outputDimension = self.__docLayerNodesNum
