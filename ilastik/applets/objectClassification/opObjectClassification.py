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
from ilastik.utility import OperatorSubView, MultiLaneOperatorABC, OpMultiLaneWrapper 
_MAXLABELS = 2

class OpObjectClassification(Operator, MultiLaneOperatorABC):
    name = "OpObjectClassification"
    category = "Top-level"

    ###############
    # Input slots #
    ###############
    BinaryImages = InputSlot(level=1) # for visualization
    SegmentationImages = InputSlot(level=1)
    ObjectFeatures = InputSlot(rtype=List, level=1)
    LabelsAllowedFlags = InputSlot(stype='bool', level=1)
    LabelInputs = InputSlot(stype=Opaque, optional=True, level=1)
    FreezePredictions = InputSlot(stype='bool')
    ObjectCounts = InputSlot(stype=Opaque, rtype=List, level=1)

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
        opkwargs = dict(parent=self)
        self.opInputShapeReader = OpMultiLaneWrapper(OpShapeReader, **opkwargs)
        self.opTrain = OpObjectTrain(graph=self.graph)
        self.opPredict = OpMultiLaneWrapper(OpObjectPredict, **opkwargs)
        self.opLabelsToImage = OpMultiLaneWrapper(OpToImage, **opkwargs)
        self.opPredictionsToImage = OpMultiLaneWrapper(OpToImage, **opkwargs)

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
        self._resizeLabelInputs(imageIndex)

    def _resizeLabelInputs(self, imageIndex, roi=None):
        #if roi is None:
        #    roi = [slice(None, None, None)]
        labels = dict()
        counts = self.ObjectCounts[imageIndex]([]).wait() # WHY cant we use .value???
        for t in counts.keys():
            # add one for background,))
            labels[t] = numpy.zeros((counts[t] + 1),)

        # FIXME: does this do the right thing?
        self.LabelInputs[imageIndex].setValue(labels)


    def setupOutputs(self):
        pass

    def setInSlot(self, slot, subindex, roi, value):
        # Nothing to do here: All inputs that support __setitem__ are
        #   directly connected to internal operators.
        pass

    def propagateDirty(self, slot, subindex, roi):
        if slot == self.ObjectCounts:
            self._resizeLabelInputs(subindex, roi)

    def addLane(self, laneIndex):
        numLanes = len(self.SegmentationImages)
        assert numLanes == laneIndex, "Image lanes must be appended."
        for slot in self.inputs.values():
            if slot.level > 0 and len(slot) == laneIndex:
                slot.resize(numLanes + 1)

    def removeLane(self, laneIndex, finalLength):
        for slot in self.inputs.values():
            if slot.level > 0 and len(slot) == finalLength + 1:
                slot.removeSlot(laneIndex, finalLength)

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)


class OpObjectTrain(Operator):
    name = "TrainRandomForestObjects"
    description = "Train a random forest on multiple images"
    category = "Learning"

    # FIXME: both of these should have rtype List. It's not used now,
    # because you can't call setValue on it (because it then calls
    # setDirty with an empty slice and fails)

    Labels = InputSlot(level=1)
    Features = InputSlot(level=1, rtype=List)
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

        for i in range(len(self.Labels)):
            # FIXME: why can't we use .value?
            labels = self.Labels[i][:].wait() # FIXME: sometimes [0]???

            for t in range(roi.start[0], roi.stop[0]):
                lab = labels[t].squeeze()
                feats = self.Features[i]([t]).wait()
                counts = feats[t][0]['Count']
                counts = numpy.asarray(counts.squeeze())
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

    Features = InputSlot(rtype=List)
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
        feats = {}
        predictions = {}
        for t in range(roi.start[0], roi.stop[0]):
            features = self.Features([t]).wait()[t][0]
            tempfeats = numpy.asarray(features['Count']).astype(numpy.float32)

            ### FIXME: is this right???
            if tempfeats.ndim == 1:
                tempfeats.resize(tempfeats.shape + (1,))
            ###

            feats[t] = tempfeats
            predictions[t]  = [0] * len(forests)

        def predict_forest(t, number):
            predictions[t][number] = forests[number].predictLabels(feats[t])

        # predict the data with all the forests in parallel
        pool = Pool()

        for t in range(roi.start[0], roi.stop[0]):
            for i, f in enumerate(forests):
                req = pool.request(partial(predict_forest, t, i))

        pool.wait()
        pool.clean()

        #FIXME we return from here for now, but afterwards we should
        #really average the pool results
        return predictions

        prediction = numpy.dstack(predictions)
        prediction = numpy.average(prediction, axis=2)
        return prediction

    def propagateDirty(self, slot, subindex, roi):
        self.Predictions.setDirty(roi)


class OpToImage(Operator):
    """Takes a segmentation image and a mapping and returns the
    mapped image.

    For instance, map prediction labels onto objects.

    """
    name = "OpToImage"
    Image = InputSlot()
    ObjectMap = InputSlot(stype=Opaque)
    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Image.meta)

    def execute(self, slot, subindex, roi, result):
        # FIXME: .value
        im = self.Image[:].wait()
        map_ = self.ObjectMap[:].wait()
        for t in range(im.shape[0]):

            # FIXME: sometimes (1, n), sometimes (n, 1)
            tmap = map_[t]

            # FIXME: why???
            if isinstance(tmap, list):
                tmap = tmap[0]

            tmap = tmap.squeeze()
            if len(tmap) != 0:
                tmap[0] = 0
                im[t] = tmap[im[t]]
        return im[roi.toSlice()]

    def propagateDirty(self, slot, subindex, roi):
        self.Output.setDirty(slice(None, None, None))
