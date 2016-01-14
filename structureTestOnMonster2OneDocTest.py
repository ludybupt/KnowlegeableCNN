from theano import tensor as T, printing
import theano
import numpy
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
from DocEmbeddingNNOneDoc import DocEmbeddingNNOneDoc
# from DocEmbeddingNNPadding import DocEmbeddingNN
from knoweagebleClassifyFlattened import CorpusReader
import cPickle
import string
import codecs

def work(argv):
	print "Started!"
	rng = numpy.random.RandomState(23455)
	sentenceWordCount = T.ivector("sentenceWordCount")
	corpus = T.matrix("corpus")
	docLabel = T.ivector('docLabel') 
	
	# for list-type data
	layer0 = DocEmbeddingNNOneDoc(corpus, sentenceWordCount, rng, wordEmbeddingDim=200, \
													 sentenceLayerNodesNum=100, \
													 sentenceLayerNodesSize=[5, 200], \
													 docLayerNodesNum=100, \
													 docLayerNodesSize=[3, 100])

	layer1 = HiddenLayer(
		rng,
		input=layer0.output,
		n_in=layer0.outputDimension,
		n_out=100,
		activation=T.tanh
	)
	
	layer2 = LogisticRegression(input=layer1.output, n_in=100, n_out=2)

	cost = layer2.negative_log_likelihood(1 - layer2.y_pred)
		
	grads = T.grad(cost, layer0.sentenceResults)
	score = T.diag(T.dot(grads, T.transpose(layer0.sentenceResults)))
	
	# construct the parameter array.
	params = layer2.params + layer1.params + layer0.params
	
	
	# Load the parameters last time, optionally.
	loadParamsVal(params)
	print "Compiling computing graph."
	output_model = theano.function(
 		[corpus, sentenceWordCount],
 		[layer2.y_pred, score]
 	)
	
	print "Compiled."
	cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset="data/train_valid/split")
	count = 0
	while(count <= 1000):
		info = cr.getCorpus([count, count + 1])
		count += 1
		if info is None:
			print "Pass"
			continue
		docMatrixes, _, sentenceWordNums, ids, sentences = info
		docMatrixes = numpy.matrix(
		            docMatrixes,
		            dtype=theano.config.floatX
		        )
		sentenceWordNums = numpy.array(
		            sentenceWordNums,
		            dtype=numpy.int32
		        )
		print "start to predict: %s." % ids[0]
		pred_y, g = output_model(docMatrixes, sentenceWordNums)
		print "End predicting."
		print "Writing resfile."
		
		score_sentence_list = zip(g, sentences)
		score_sentence_list.sort(key=lambda x:-x[0])
		
		with codecs.open("data/output/" + str(pred_y[0]) + "/" + ids[0], "w", 'utf-8', "ignore") as f:
			f .write("pred_y: %i\n" % pred_y[0])
			for g0, s in score_sentence_list:
				f.write("%f\t%s\n" % (g0, string.join(s, " ")))
# 		print zip(ids, pred_y[0])
# 		f = file("data/test/res/res" + str(count), "w")
# 		f.write(str(zip(ids, pred_y[0])))
# 		f.close()
		print "Written." + str(count)
		
		
	print "All finished!"
	
def saveParamsVal(params):
	with open("model/scnn.model", 'wb') as f:  # open file with write-mode
		for para in params:
			cPickle.dump(para.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)  # serialize and save object

def loadParamsVal(params):
	try:
		with open("model/scnn.model", 'rb') as f:  # open file with write-mode
			for para in params:
				para.set_value(cPickle.load(f), borrow=True)
	except:
		pass
	
def transToTensor(data, t):
	return theano.shared(
        numpy.array(
            data,
            dtype=t
        ),
        borrow=True
    )
if __name__ == '__main__':
	work("train")
