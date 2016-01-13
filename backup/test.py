import theano
import theano.typed_list as ttList
from theano import tensor as T, function, printing
import theano.tensor.signal.conv
import numpy
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.type import TensorType
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
from DocEmbeddingNN import DocEmbeddingNN

def dealWithSentence(rng, sentence):
    sentenceW = theano.shared(
        numpy.asarray(
            rng.uniform(low=-0.2, high=0.2, size=(2, 2, 2)),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    sentence_out = theano.tensor.signal.conv.conv2d(input=sentence, filters=sentenceW)
#     sentence_pool = T.max(sentence_out, axis = [1])
    sentence_pool = theano.tensor.signal.downsample.max_pool_2d(sentence_out, (1000, 1))
    sentence_embedding = sentence_pool.reshape((1, 2 * 3))
    return sentence_embedding
def dealWithOneDoc(rng, doc):
    sentenceList = doc
    sentenceSize = ttList.length(sentenceList)
    sentenceResults, _ = theano.scan(fn=lambda i, sl: dealWithSentence(rng, sl[i]),
                        non_sequences=[sentenceList],
                         sequences=[T.arange(sentenceSize, dtype='int64')])
    sentenceResults = sentenceResults.dimshuffle([0, 2])
    
    docW = theano.shared(
        numpy.asarray(
            rng.uniform(low=-0.2, high=0.2, size=(2, 1, 2)),
            dtype=theano.config.floatX
        ),
        borrow=True
    )
    
    doc_out = theano.tensor.signal.conv.conv2d(input=sentenceResults, filters=docW)
    
#     docPool = T.max(doc_out, axis = [0])
    docPool = theano.tensor.signal.downsample.max_pool_2d(doc_out, (1000, 1))
    doc_embedding = T.reshape(docPool, (1, 2 * 5)).dimshuffle([1])
    
#     doc_embeddingp = printing.Print('doc_embedding')
#     doc_embedding = doc_embeddingp(doc_embedding)
    
    return doc_embedding
def main():
    rng = numpy.random.RandomState(23455)
    a = [
                 [
                  [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                  [[1, 2, 3, 4], [1, 2, 3, 4]]
                 ]
                ]


    docList = ttList.TypedListType(ttList.TypedListType(T.fmatrix))("docList")
    docSize = ttList.length(docList)
    
    modelResults, _ = theano.scan(fn=lambda i, tl: dealWithOneDoc(rng, tl[i]),
                        non_sequences=[docList],
                         sequences=[T.arange(docSize, dtype='int64')])
    
    testFunc = theano.function([docList], modelResults);
    rrrr = testFunc(a)
    print rrrr[0]
    print "All finished!"

def main2():
    a = [
            [
                [[1, 2, 3]], [[4, 5]]
             ]
         ]
    tl = ttList.TypedListType(ttList.TypedListType(theano.tensor.fmatrix))()
    o = ttList.length(tl)
    f = theano.function([tl], o)
    testRes = f(a)
    print testRes
    print "All finished!"
    
def main3():
    rng = numpy.random.RandomState(23455)
 
    docList = ttList.TypedListType(ttList.TypedListType(TensorType(theano.config.floatX, (False, False))))("docList")
    docLabel = T.ivector('docLabel') 
    layer0 = DocEmbeddingNN(docList, rng, 4)
    
    layer1 = HiddenLayer(
        rng,
        input=layer0.output,
        n_in=layer0.outputDimension,
        n_out=10,
        activation=T.tanh
    )
    
    layer2 = LogisticRegression(input=layer1.output, n_in=10, n_out=10)
    cost = layer2.negative_log_likelihood(docLabel)
    params = layer2.params + layer1.params + layer0.params
    grads = T.grad(cost, params)
    
    f = theano.function([docList], layer2.y_pred)
    
    
    a = [
            [
              [[2, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
              [[1, 2, 4, 4], [1, 2, 3, 4]]
             ],
          [
              [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
              [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
             ]
        ]
    print f(a)
    print "All finished!"
def main4():
    a = [
            [
              [[2, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
              [[1, 2, 4, 4], [1, 2, 3, 4]]
             ],
          [
              [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
              [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
             ]
        ]
    b = numpy.matrix(a)
    w = T.fvector("W")
    l = T.ivector("l")
    d, _ = theano.scan(fn=lambda i, sl, w: T.dot(w, sl[i]),
                            non_sequences=[l, w],
                             sequences=[T.arange(l[0], dtype='int64')])
    sumx = d.sum();
    g = T.grad(sumx, w)
    f = theano.function([w, l], g)
    
def main5():
    docs = T.ftensor4("docs")
    dsnv = T.fvector("dsn")
    swnm = T.fmatrix("swn")
    dw = T.fmatrix("dw")
    sw = T.fmatrix("sw")

    
    def localConv(doc, dsn, swnv, dww, sww):
#         t = T.arange(docSentenceSize)
#         ccc = docs[t.nonzero()]
       
        t = T.arange(dsn).nonzero()

        t = (T.arange(10000) < dsn).nonzero()
#         print t
#         t=T.arange(dsn)
        docSub = doc[t]
        p = printing.Print('docSub')
        docSub = p(docSub)
        swnvSub = swnv[t]
        def sentenceConv(sen, wn, sww):
            t = (T.arange(10000) < wn).nonzero()
            senSub = sen[t];
            convRes = theano.tensor.signal.conv.conv2d(senSub, sww)
            sentence_pool = theano.tensor.signal.downsample.max_pool_2d(convRes, (100000, 1)).flatten(1)
            return sentence_pool
            
        sentenceLayer, _ = theano.scan(fn=lambda sen, wn, sww: sentenceConv(sen, wn, sww),
                                                            non_sequences=[sww],
                                                            sequences=[docSub, swnvSub])

        
        convRes = theano.tensor.signal.conv.conv2d(sentenceLayer, dww)
        
        sentence_pool = theano.tensor.signal.downsample.max_pool_2d(convRes, (100000, 1)).flatten(1)
        return sentence_pool
    
    res, _ = theano.scan(fn=lambda doc, dsn, swnv, dww, sww: localConv(doc, dsn, swnv, dww, sww),
                        non_sequences=[dw, sw],
                        sequences=[docs, dsnv, swnm]
                         )
    
#     p = printing.Print('res')
#     res = p(res)
    cost = res.sum()
    g = T.grad(cost, [dw, sw])
    
    
    f = theano.function([docs, dsnv, swnm, dw, sw], g)
    
    d = [
            [
                 [
                    [2, 2, 3, 4],
                    [1, 2, 3, 4],
                    [3, 1, 2, 3],
                    [6, 4, 2, 1],
                    [0, 0, 0, 0]
                ],
                 [
                    [4, 3, 2, 1],
                    [4, 6, 9, 2],
                    [6, 6, 3, 1],
                    [2, 5, 2, 9],
                    [3, 2, 1, 7]
                ]
             ],
            [
                     [
                        [9, 8, 7, 6],
                        [5, 4, 3, 2],
                        [1, 9, 8, 7],
                        [6, 5, 4, 3],
                        [0, 0, 0, 0]
                    ],
                     [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ]
                 ]
            ]
    
    docSentenceCount = [2, 1]
    sentenceWordCount = [[4, 5], [4, 0]]
    docF = [[1, 1]]
    senF = [[1, 2], [1, 2]]
    print f(d, docSentenceCount, sentenceWordCount, docF, senF)
    
    
    print "All finished!"

def main6():
    print "Start!"
    d = [
            [
                 [
                    [2, 2, 3, 4],
                    [1, 2, 3, 4],
                    [3, 1, 2, 3],
                    [6, 4, 2, 1],
                    [0, 0, 0, 0]
                ],
                 [
                    [4, 3, 2, 1],
                    [4, 6, 9, 2],
                    [6, 6, 3, 1],
                    [2, 5, 2, 9],
                    [3, 2, 1, 7]
                ]
             ],
            [
                     [
                        [9, 8, 7, 6],
                        [5, 4, 3, 2],
                        [1, 9, 8, 7],
                        [6, 5, 4, 3],
                        [0, 0, 0, 0]
                    ],
                     [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ]
                 ]
            ]
    mr = numpy.max([numpy.max([2, 1]), numpy.max([[4, 5], [4, 0]])])
    
#     docSentenceCount = [2, 1]
#     sentenceWordCount = [[4, 5], [4, 0]]
    docSentenceCount = T.fvector("docSentenceCount")
    sentenceWordCount = T.fmatrix("sentenceWordCount")
    corpus = T.ftensor4("corpus")
    rng = numpy.random.RandomState(23455)
    
    
    layer0 = DocEmbeddingNN(corpus, docSentenceCount, sentenceWordCount, rng, 4, mr, 3, [2, 2], 2, [1, 2])
    s = layer0.output.sum()
    g = theano.grad(s, layer0.params)
    print "Compiling!"
    f = theano.function([corpus, docSentenceCount, sentenceWordCount], g)
    print "Compiled!"
    print f(d, [2, 1], [[4, 5], [4, 0]])
    print "All finished!"
if __name__ == '__main__':
    main6()
