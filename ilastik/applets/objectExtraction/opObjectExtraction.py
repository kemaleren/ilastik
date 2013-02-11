import numpy
import h5py
import vigra.analysis

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.rtype import SubRegion, List

class OpLabelImage(Operator):
    name = "Label Image Accessor"
    BinaryImage = InputSlot()

    # List of background label in the order of channels of the binary
    # image; -1 if the label image should not be computed for that
    # channel
    BackgroundLabels = InputSlot(optional=True)
    defaultBackground = 0

    LabelImage = OutputSlot()

    # Pull the following output slot to compute the label image. This
    # slot is introduced in order to do a precomputation of the output
    # image and to write the results into a compressed hdf5 virtual
    # file instead of allocating space for all the requests.
    LabelImageComputation = OutputSlot(stype="float")

    def __init__(self, parent=None, graph=None):
        super(OpLabelImage, self).__init__(parent=parent,graph=graph)
        self._processedTimeSteps = []

    def setupOutputs(self):
        self.LabelImage.meta.assignFrom(self.BinaryImage.meta)
        self.LabelImage.meta.dtype = numpy.uint32
        self.LabelImageComputation.meta.dtype = numpy.float
        self.LabelImageComputation.meta.shape = [0]

        shape = self.LabelImage.meta.shape
        self._label_img = numpy.zeros(shape, dtype=numpy.uint32)

    def execute(self, slot, subindex, roi, destination):
        if slot is self.LabelImage:
            for t in range(roi.start[0],roi.stop[0]):
                slc = roi.toSlice()
                slc = (slice(t, t + 1),) + slc[1:]
                if t not in self._processedTimeSteps:
                    destination[t-roi.start[0]:t-roi.start[0]+1,...] = 0
                else:
                    destination[t-roi.start[0]:t-roi.start[0]+1,...] = self._label_img[slc]
            return destination

        if slot is self.LabelImageComputation:
            channels = self.BinaryImage.meta.shape[-1]
            for t in range(roi.start[0], roi.stop[0]):
                if t not in self._processedTimeSteps:
                    for ch in range(channels):
                        print "Calculating LabelImage at t=" + str(t) + ", ch=" + str(ch) + " "
                        sroi = SubRegion(self.BinaryImage, start=[t,0,0,0,ch],
                                         stop=[t+1,] + list(self.BinaryImage.meta.shape[1:-1]) + [ch+1,])
                        a = self.BinaryImage.get(sroi).wait()
                        a = numpy.array(a[0,...,0],dtype=numpy.uint8)
                        if self.BackgroundLabels.ready():
                            backgroundLabel = self.BackgroundLabels.value[ch]
                        else:
                            backgroundLabel = self.defaultBackground
                        if backgroundLabel != -1:
                            result = vigra.analysis.labelVolumeWithBackground(a, background_value=backgroundLabel)
                            self._label_img[t, ..., ch] = result


                    self._processedTimeSteps.append(t)

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.BinaryImage:
            self.LabelImage.setDirty(roi)
        elif slot is self.BackgroundLabels:
            self.LabelImage.setDirty(roi)
        else:
            print "Unknown dirty input slot: " + str(slot.name)


def opFeaturesFactory(name, features):
    """An operator factory producing an operator that calculates a
    specific set of features.

    """
    class cls(Operator):
        LabelImage = InputSlot()
        Output = OutputSlot(stype=Opaque, rtype=List)

        def __init__(self, parent=None, graph=None):
            super(cls, self).__init__(parent=parent,
                                      graph=graph)
            self._cache = {}
            self.fixed = False

        def setupOutputs(self):
            # number of time steps
            self.Output.meta.shape = self.LabelImage.meta.shape[0:1]
            self.Output.meta.dtype = object

        @staticmethod
        def extract(a):
            labels = numpy.asarray(a, dtype=numpy.uint32)
            data = numpy.asarray(a, dtype=numpy.float32)
            feats = vigra.analysis.extractRegionFeatures(data,
                                                         labels,
                                                         features=features,
                                                         ignoreLabel=0)
            return feats

        def execute(self, slot, subindex, roi, result):
            if slot is not self.Output:
                return
            feats = {}
            if len(roi) == 0:
                roi = range(self.LabelImage.meta.shape[0])
            for t in roi:
                if t in self._cache:
                    feats_at = self._cache[t]
                elif self.fixed:
                    feats_at = dict((f, numpy.asarray([[]])) for f in features)
                else:
                    feats_at = []
                    lshape = self.LabelImage.meta.shape
                    numChannels = lshape[-1]
                    for c in range(numChannels):
                        tcroi = SubRegion(self.LabelImage,
                                          start = [t,] + (len(lshape) - 2) * [0,] + [c,],
                                          stop = [t+1,] + list(lshape[1:-1]) + [c+1,])
                        a = self.LabelImage.get(tcroi).wait()
                        a = a[0,...,0] # assumes t,x,y,z,c
                        feats_at.append(self.extract(a))
                    self._cache[t] = feats_at
                feats[t] = feats_at
            return feats

        def propagateDirty(self, slot, subindex, roi):
            if slot is self.LabelImage:
                self.Output.setDirty(List(self.Output,
                                          range(roi.start[0], roi.stop[0])))

    cls.__name__ = name
    return cls

OpRegionFeatures = opFeaturesFactory('OpRegionFeatures',
                                     ['Count',
                                      'RegionCenter',
                                      'Coord<ArgMaxWeight>',
                                      'Coord<Minimum>',
                                      'Coord<Maximum>',
                                  ])


class OpObjectCenterImage(Operator):
    """A cross in the center of each connected component."""
    BinaryImage = InputSlot()
    RegionCenters = InputSlot(rtype=List)
    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.BinaryImage.meta)

    @staticmethod
    def __contained_in_subregion(roi, coords):
        b = True
        for i in range(len(coords)):
            b = b and (roi.start[i] <= coords[i] and coords[i] < roi.stop[i])
        return b

    @staticmethod
    def __make_key(roi, coords):
        key = [coords[i] - roi.start[i] for i in range(len(roi.start))]
        return tuple(key)

    def execute(self, slot, subindex, roi, result):
        result[:] = 0
        for t in range(roi.start[0], roi.stop[0]):
            centers = self.RegionCenters([t]).wait()
            for ch in range(roi.start[-1], roi.stop[-1]):
                centers = centers[t][ch]['RegionCenter']
                centers = numpy.asarray(centers, dtype=numpy.uint32)
                if centers.size:
                    centers = centers[1:,:]
                for center in centers:
                    x, y, z = center[0:3]
                    for dim in (1, 2, 3):
                        for offset in (-1, 0, 1):
                            c = [t, x, y, z, ch]
                            c[dim] += offset
                            c = tuple(c)
                            if self.__contained_in_subregion(roi, c):
                                result[self.__make_key(roi, c)] = 255
        return result

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.RegionCenters:
            # FIXME: proper dirty propagation.
            self.Output.setDirty([])


class OpObjectExtraction(Operator):
    name = "Object Extraction"

    RawImage = InputSlot()
    BinaryImage = InputSlot()
    BackgroundLabels = InputSlot()

    LabelImage = OutputSlot()
    ObjectCenterImage = OutputSlot()
    RegionCenters = OutputSlot(stype=Opaque, rtype=List)
    RegionFeatures = OutputSlot(stype=Opaque, rtype=List)

    def __init__(self, parent=None, graph=None):

        super(OpObjectExtraction, self).__init__(parent=parent,
                                                 graph=graph)

        # internal operators
        self._opLabelImage = OpLabelImage(parent=self, graph = graph)
        self._opRegFeats = OpRegionFeatures(parent=self, graph = graph)
        self._opObjectCenterImage = OpObjectCenterImage(parent=self, graph=self.graph)

        # connect internal operators
        self._opLabelImage.BinaryImage.connect(self.BinaryImage)
        self._opLabelImage.BackgroundLabels.connect(self.BackgroundLabels)

        self._opRegFeats.LabelImage.connect(self._opLabelImage.LabelImage)

        self._opObjectCenterImage.BinaryImage.connect(self.BinaryImage)
        self._opObjectCenterImage.RegionCenters.connect(self._opRegFeats.Output)

        # connect outputs
        self.LabelImage.connect(self._opLabelImage.LabelImage)
        self.ObjectCenterImage.connect(self._opObjectCenterImage.Output)
        self.RegionFeatures.connect(self._opRegFeats.Output)

    def setupOutputs(self):
        pass

    def execute(self, slot, subindex, roi, result):
        pass

    def propagateDirty(self, inputSlot, subindex, roi):
        pass
