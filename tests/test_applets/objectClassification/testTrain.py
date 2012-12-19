import vigra
import numpy
import h5py
import ilastik.applets.objectClassification.opObjectClassification
print ilastik.applets.objectClassification.opObjectClassification.__file__
#from ilastik.applets.objectClassification.opObjectClassification import OpObjectTrain, OpObjectPredict
from lazyflow.request import Request, Pool
from lazyflow.graph import Graph

def simpleTest():
    ccfile = h5py.File("/home/akreshuk/data/circles3d_cc.h5")
    cc = ccfile["/volume/data"]

    maxobject = numpy.max(cc)

    cc =numpy.squeeze(cc)
    cc = numpy.asarray(cc, dtype=numpy.uint32)
    data = numpy.asarray(cc, dtype=numpy.float32)

    print cc.shape
    print cc.dtype, data.dtype

    #comput features
    feats = vigra.analysis.extractRegionFeatures(data, cc, features=['RegionCenter', 'Count'], ignoreLabel=0)
    print feats
    counts = numpy.asarray(feats['Count'])
    counts = counts[1:]
    print counts

    #generate labels
    labels = numpy.zeros((counts.shape[0],), dtype=numpy.uint32)
    labels[0] = 1
    labels[1] = 1
    labels[2] = 2
    labels[-1] = 2
    print labels

    #make a matrix
    index = numpy.nonzero(labels)
    newlabels = labels[index]
    newfeats = counts[index]

    print newlabels.shape, newfeats.shape
    print newlabels.dtype, newfeats.dtype

    newlabels.resize(newlabels.shape+(1,))
    newfeats.resize(newfeats.shape+(1,))

    print newlabels.shape, newlabels.dtype, newlabels
    print newfeats.shape, newfeats.dtype, newfeats

    rf= vigra.learning.RandomForest()
    oob = rf.learnRF(newfeats.astype(numpy.float32), newlabels)
    print oob

    counts.resize(counts.shape+(1,))
    pred = rf.predictLabels(counts.astype(numpy.float32))
    print pred

def operatorTest():
    
    graph = Graph()
    
    ccfile = h5py.File("/home/mschiegg/data/circles3d_cc.h5")
    cc = ccfile["/volume/data"]
    cc =numpy.squeeze(cc)
    cc = numpy.asarray(cc, dtype=numpy.uint32)
    data = numpy.asarray(cc, dtype=numpy.float32)

    #compute features
    feats = vigra.analysis.extractRegionFeatures(data, cc, features=['RegionCenter', 'Count'], ignoreLabel=0)
    print feats
    counts = numpy.asarray(feats['Count'])
    counts = counts[1:]
    print counts
    counts.resize(counts.shape+(1,))
    
    
    #generate labels
    labels = numpy.zeros((counts.shape[0],), dtype=numpy.uint32)
    labels[0] = 1
    labels[1] = 1
    labels[2]= 2
    labels[-1] = 2
    print labels
    
    opTrain = ilastik.applets.objectClassification.opObjectClassification.OpObjectTrain(graph=graph)
    print opTrain
    opTrain.Labels.resize(1)
    opTrain.Features.resize(1)
    opTrain.Labels[0].setValue(labels)
    opTrain.Features[0].setValue(counts)
    #opTrain.Labels.setValue([labels])
    #opTrain.Features.setValue([counts])
    opTrain.FixClassifier.setValue(False)
    
    print "features:", opTrain.Features[0].value
    print "are we ready to train?", opTrain.Labels.ready(), opTrain.Features.ready(), opTrain.FixClassifier.ready(), opTrain.Classifier.ready()
    
    cl = opTrain.Classifier[:].wait()
    print cl
    

    opPredict = OpObjectPredict(graph=graph)
    #opPredict.Features.resize(1)
    opPredict.Features.setValue(counts)
    opPredict.LabelsCount.setValue(2)
    opPredict.Classifier.setValue(cl)
    
    print "are we ready to predict?", opPredict.Features.ready(), opPredict.LabelsCount.ready(), opPredict.Classifier.ready(), opPredict.Predictions.ready()
   
    
    preds = opPredict.Predictions[:].wait()
    print preds

    
    
if __name__=='__main__':
    print "madnesss"
    operatorTest()