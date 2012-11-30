import numpy
import h5py
import vigra
import vigra.analysis
import copy
from collections import defaultdict

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.rtype import Everything, SubRegion, List
from lazyflow.operators.ioOperators.opStreamingHdf5Reader import OpStreamingHdf5Reader
from lazyflow.operators.ioOperators.opInputDataReader import OpInputDataReader
from lazyflow.operators import OperatorWrapper, OpBlockedSparseLabelArray, OpValueCache, \
OpMultiArraySlicer2, OpSlicedBlockedArrayCache, OpPrecomputedInput
from lazyflow.request import Request, Pool
from functools import partial

from ilastik.applets.pixelClassification.opPixelClassification import OpShapeReader, OpMaxValue

_MAXLABELS = 2


class OpObjectClassification(Operator):
    name = "OpObjectClassification"
    category = "Top-level"

    # FIXME: labels, features, etc. depend on time slice.

    ###############
    # Input slots #
    ###############
    BinaryImages = InputSlot(level=1) # for visualization
    SegmentationImages = InputSlot(level=1)
    ObjectFeatures = InputSlot(stype=Opaque, rtype=List, level=1)
    LabelsAllowedFlags = InputSlot(stype='bool', level=1)
    LabelInputs = InputSlot(stype=Opaque, optional=True, level=1)
    FreezePredictions = InputSlot(stype='bool')
    ObjectCounts = InputSlot(level=1, stype=Opaque)

    ################
    # Output slots #
    ################
    NumLabels = OutputSlot()
    Classifier = OutputSlot()
    LabelImages = OutputSlot(level=1)
    PredictionImages = OutputSlot(level=1)
    SegmentationImagesOut = OutputSlot(level=1)

    # FIXME: not actually used
    Eraser = OutputSlot()
    DeleteLabel = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpObjectClassification, self).__init__(*args, **kwargs)

        # internal operators
        opkwargs = dict(parent=self, graph=self)
        self.opInputShapeReader = OperatorWrapper(OpShapeReader, **opkwargs)
        self.opTrain = OpObjectTrain(graph=self.graph)
        self.opPredict = OperatorWrapper(OpObjectPredict, **opkwargs)
        self.opLabelsToImage = OperatorWrapper(OpToImage, **opkwargs)
        self.opPredictionsToImage = OperatorWrapper(OpToImage, **opkwargs)

        # connect inputs
        self.opInputShapeReader.Input.connect(self.SegmentationImages)

        self.opTrain.inputs["Features"].connect(self.ObjectFeatures)
        self.opTrain.inputs['Labels'].connect(self.LabelInputs)
        self.opTrain.inputs['FixClassifier'].setValue(False)

        self.opPredict.inputs["Features"].connect(self.ObjectFeatures)
        self.opPredict.inputs["Classifier"].connect(self.opTrain.outputs["Classifier"])
        self.opPredict.inputs["LabelsCount"].setValue(_MAXLABELS)

        self.opLabelsToImage.inputs["Image"].connect(self.SegmentationImages)
        self.opLabelsToImage.inputs["ObjectMap"].connect(self.LabelInputs)

        self.opPredictionsToImage.inputs["Image"].connect(self.SegmentationImages)
        self.opPredictionsToImage.inputs["ObjectMap"].connect(self.opPredict.Predictions)

        # connect outputs
        self.NumLabels.setValue(_MAXLABELS)
        self.LabelImages.connect(self.opLabelsToImage.Output)
        self.PredictionImages.connect(self.opPredictionsToImage.Output)
        self.Classifier.connect(self.opTrain.Classifier)

        self.SegmentationImagesOut.connect(self.SegmentationImages)

        # TODO: remove these
        self.Eraser.setValue(100)
        self.DeleteLabel.setValue(-1)

        def handleNewInputImage(multislot, index, *args):
            def handleInputReady(slot):
                self.setupCaches(multislot.index(slot))
            multislot[index].notifyReady(handleInputReady)

        self.SegmentationImages.notifyInserted(handleNewInputImage)

    def setupCaches(self, imageIndex):
        """Setup the label input to correct dimensions"""
        numImages=len(self.SegmentationImages)
        self.LabelInputs.resize(numImages)
        self.LabelInputs[imageIndex].meta.shape = (1,)
        self.LabelInputs[imageIndex].meta.dtype = object
        self.LabelInputs[imageIndex].meta.axistags = None

        d = defaultdict(lambda: numpy.zeros((0,)))
        self.LabelInputs[imageIndex].setValue(d)

    def setupOutputs(self):
        pass

    def setInSlot(self, slot, subindex, roi, value):
        # Nothing to do here: All inputs that support __setitem__ are
        #   directly connected to internal operators.
        pass

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.ObjectCounts:
            self._resizeLabelInputs(subindex, roi)

    def _resizeLabelInputs(self, imageIndex, roi=None):
        labels = self.LabelInputs[imageIndex][roi]
        counts = self.ObjectCounts[imageIndex][roi]
        tstart, tstop = roi.start[0], roi.stop[0]
        for t in range(tstart, tstop):
            nlabels = len(labels[t])
            nobjects = counts[t]
            if nlabels != nobjects:
                labels[t] = numpy.zeros((nobjects,))
                self.LabelInputs[imageIndex][t] = numpy.zeros((nobjects,))


class OpObjectTrain(Operator):
    name = "TrainRandomForestObjects"
    description = "Train a random forest on multiple images"
    category = "Learning"

    # FIXME: both of these should have rtype List. It's not used now,
    # because you can't call setValue on it (because it then calls
    # setDirty with an empty slice and fails)

    Labels = InputSlot(level=1)
    Features = InputSlot(stype=Opaque, level=1, rtype=List)
    FixClassifier = InputSlot(stype="bool")

    Classifier = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpObjectTrain, self).__init__(*args, **kwargs)
        self._forest_count = 1
        # TODO: Make treecount configurable via an InputSlot
        self._tree_count = 100
        self.FixClassifier.setValue(False)

    def setupOutputs(self):
        if self.inputs["FixClassifier"].value == False:
            self.outputs["Classifier"].meta.dtype = object
            self.outputs["Classifier"].meta.shape = (self._forest_count,)
            self.outputs["Classifier"].meta.axistags  = None

    def execute(self, slot, subindex, roi, result):
        featMatrix = []
        labelsMatrix = []

        for i, labels in enumerate(self.Labels):
            feats = self.Features[i][0].wait()
            counts = numpy.asarray(feats[0]['Count']).squeeze()
            lab = labels[:].wait()[0].squeeze()
            index = numpy.nonzero(lab)
            featMatrix.append(counts[index])
            labelsMatrix.append(lab[index])

        if len(featMatrix) == 0 or len(labelsMatrix) == 0:
            result[:] = None
        else:
            featMatrix = numpy.concatenate(featMatrix, axis=0).reshape(-1, 1)
            labelsMatrix = numpy.concatenate(labelsMatrix, axis=0).reshape(-1, 1)
            try:
                # train and store forests in parallel
                pool = Pool()
                for i in range(self._forest_count):
                    def train_and_store(number):
                        result[number] = vigra.learning.RandomForest(self._tree_count)
                        result[number].learnRF(featMatrix.astype(numpy.float32),
                                               labelsMatrix.astype(numpy.uint32))
                    req = pool.request(partial(train_and_store, i))
                pool.wait()
                pool.clean()
            except:
                print ("couldn't learn classifier")
                raise
        return result

    def propagateDirty(self, slot, subindex, roi):
        if slot is not self.FixClassifier and \
           self.inputs["FixClassifier"].value == False:
            self.outputs["Classifier"].setDirty((slice(0,1,None),))


class OpObjectPredict(Operator):
    name = "OpObjectPredict"

    Features = InputSlot(stype=Opaque, rtype=List)
    LabelsCount = InputSlot(stype='integer')
    Classifier = InputSlot()

    Predictions = OutputSlot(stype=Opaque)

    def setupOutputs(self):

        self.Predictions.meta.shape=(1,)
        self.Predictions.meta.dtype = object
        self.Predictions.meta.axistags = None

    def execute(self, slot, subindex, roi, result):

        forests=self.inputs["Classifier"][:].wait()

        if forests is None:
            # Training operator may return 'None' if there was no data
            # to train with
            return numpy.zeros(numpy.subtract(roi.stop, roi.start),
                               dtype=numpy.float32)[...]

        # FIXME FIXME: over here, we should really select only the
        # objects in the roi. However, roi of list type doesn't work
        # with setValue, so for now we compute everything.
        features = self.Features[0].wait()
        counts = numpy.asarray(features[0]['Count'])
        if len(counts.shape)==1:
            counts.resize(counts.shape+(1,))

        predictions = [0]*len(forests)

        def predict_forest(number):
            predictions[number] = forests[number].predictLabels(counts.astype(numpy.float32))

        # predict the data with all the forests in parallel
        pool = Pool()

        for i,f in enumerate(forests):
            req = pool.request(partial(predict_forest, i))

        pool.wait()
        pool.clean()

        #FIXME we return from here for now, but afterwards we should
        #really average the pool results
        return predictions

        prediction=numpy.dstack(predictions)
        prediction = numpy.average(prediction, axis=2)
        return prediction

    def propagateDirty(self, slot, subindex, roi):
        self.Predictions.setDirty(roi)


class OpToImage(Operator):
    name = "OpToImage"
    Image = InputSlot()
    ObjectMap = InputSlot(stype=Opaque)
    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Image.meta)

    def execute(self, slot, subindex, roi, result):
        im = self.Image[:].wait()
        map_ = self.ObjectMap[:].wait()
        map_ = map_[0].squeeze()
        if len(map_) != 0:
            map_[0] = 0
            im = map_[im]
        return im[roi.toSlice()]

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None, None, None))


class OpMaxListValue(Operator):
    #For each image, the input is a relabeling vector
    Inputs = InputSlot(level=1) # A list of lists of values
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpMaxListValue, self).__init__(*args, **kwargs)
        self.Output.meta.shape = (1,)
        self.Output.meta.dtype = object
        self._output = 0

    def setupOutputs(self):
        self.updateOutput()
        self.Output.setValue(self._output)

    def execute(self, slot, subindex, roi, result):
        result[0] = self._output
        return result

    def propagateDirty(self, inputSlot, subindex, roi):
        self.updateOutput()
        self.Output.setValue(self._output)

    def updateOutput(self):
        # Return the max value of all our inputs
        maxValue = None
        for i, inputSubSlot in enumerate(self.Inputs):
            # Only use inputs that are actually configured
            if inputSubSlot.ready():
                if len(inputSubSlot.value)>0:
                    curMax = numpy.max(inputSubSlot.value)
                    if maxValue is None:
                        maxValue = curMax
                    else:
                        maxValue = max(maxValue, curMax)

        if maxValue is None:
            self._output = 0
        else:
            self._output = maxValue