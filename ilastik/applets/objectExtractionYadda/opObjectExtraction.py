import numpy
import h5py
import vigra
import vigra.analysis

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.rtype import Everything, SubRegion, List
from lazyflow.operators.ioOperators.opStreamingHdf5Reader import OpStreamingHdf5Reader

from collections import defaultdict

class OpObjectExtraction(Operator):
    name = "Object Extraction"

    BinaryImage = InputSlot()

    SegmentationImage = OutputSlot()
    ObjectCenterImage = OutputSlot()

    # all of these slots produce a dictionary keyed by integers
    # indexing the time dimension.
    RegionFeatures = OutputSlot(stype=Opaque, rtype=List)
    RegionCenters = OutputSlot(stype=Opaque, rtype=List)
    ObjectCounts = OutputSlot(stype=Opaque, rtype=List)

    def __init__(self, parent = None, graph = None):
        super(OpObjectExtraction, self).__init__(parent=parent,graph=graph)

        # internal operators #
        self._opSegmentationImage = OpSegmentationImage(parent=self, graph = self.graph)

        self._opRegionCenters = OpRegionCenters(parent=self, graph=self.graph)
        self._opRegionFeats = OpRegionFeatures(parent=self, graph=self.graph)

        self._opObjectCenterImage = OpObjectCenterImage(parent=self, graph=self.graph)
        self._opObjCounts = OpObjectCounts(parent=self, graph=self.graph)

        # connect internal operators
        self._opSegmentationImage.Input.connect(self.BinaryImage)

        self._opRegionFeats.Input.connect(self._opSegmentationImage.Output)
        self._opRegionCenters.Input.connect(self._opSegmentationImage.Output)

        self._opObjectCenterImage.BinaryImage.connect(self.BinaryImage)
        self._opObjectCenterImage.RegionCenters.connect(self._opRegionCenters.Output)

        self._opObjCounts.Input.connect(self._opSegmentationImage.Output)

        # connect outputs to inner operators
        self.SegmentationImage.connect(self._opSegmentationImage.Output)
        self.ObjectCenterImage.connect(self._opObjectCenterImage.Output)
        self.RegionFeatures.connect(self._opRegionFeats.Output)
        self.ObjectCounts.connect(self._opObjCounts.Output)


    def setupOutputs(self):
        pass

    def execute(self, slot, subindex, roi, result):
        pass

    def propagateDirty(self, inputSlot, subindex, roi):
        pass


class OpSegmentationImage(Operator):
    """Perform connected component extraction"""
    Input = InputSlot()
    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

    def execute(self, slot, subindex, roi, destination):
        if slot is self.Output:
            a = self.Input.get(roi).wait()
            assert a.ndim == 5
            assert(a.shape[-1] == 1)

            # FIXME: time start may not be 0
            for t in range(a.shape[0]):
                destination[t, ..., 0] = vigra.analysis.labelVolumeWithBackground(a[t, ..., 0])
            return destination

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.Input:
            self.Output.setDirty([])


def opFeaturesFactory(name, features):
    """An operator factory producing an operator that calculates a
    specific set of features.

    """
    class cls(Operator):
        Input = InputSlot()
        Output = OutputSlot(stype=Opaque, rtype=List)

        def __init__(self, parent=None, graph=None):
            super(cls, self).__init__(parent=parent,
                                      graph=graph)
            self._cache = {}
            self.fixed = True

        def setupOutputs(self):
            # number of time steps
            self.Output.meta.shape = self.Input.meta.shape[0:1]
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
            for t in roi:
                if t in self._cache:
                    feats_at = self._cache[t]
                elif self.fixed:
                    feats_at = dict((f, numpy.asarray([[]])) for f in featuers)
                else:
                    feats_at = []
                    lshape = self.Input.meta.shape
                    numChannels = lshape[-1]
                    for c in range(numChannels):
                        tcroi = SubRegion(self.Input,
                                          start = [t,] + (len(lshape) - 2) * [0,] + [c,],
                                          stop = [t+1,] + list(lshape[1:-1]) + [c+1,])
                        a = self.Input.get(tcroi).wait()
                        a = a[0,...,0] # assumes t,x,y,z,c
                        feats_at.append(extract(a))
                    self._cache[t] = feats_at
                feats[t] = feats_at
            return feats

        def propagateDirty(self, slot, subindex, roi):
            if slot is self.Input:
                self.Output.setDirty(List(self.Output,
                                          range(roi.start[0], roi.stop[0])))

    cls.__name__ = name
    return cls

OpRegionCenters = opFeaturesFactory('OpRegionCenters', 'RegionCenter')
OpRegionFeatures = opFeaturesFactory('OpRegionFeatures',
                                     ['Count',
                                      'RegionCenter',
                                      'Coord<ArgMaxWeight>'])

class OpObjectCounts(Operator):
    """Number of objects in each image"""
    Input = InputSlot()
    Output = OutputSlot(stype=Opaque, rtype=List)

    def __init__(self, parent=None, graph=None):
        super(OpObjectCounts, self).__init__(parent=parent,
                                              graph=graph)
        def setshape(s):
            s.meta.shape = (1,)
            s.meta.dtype = object
            s.meta.axistags = None

        setshape(self.Output)

    def setupOutputs(self):
        pass

    def execute(self, slot, subindex, roi, result):
        if slot is self.Output:
            result = {}
            img = self.Input.value
            for t, img in enumerate(img):
                result[t] = img.max()
        return result

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.Input:
            self.Output.setDirty(List(slot, range(roi.start[0], roi.stop[0])))


class OpObjectCenterImage(Operator):
    """A cross in the center of each connected component."""
    BinaryImage = InputSlot()
    RegionCenters = InputSlot()
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
        tstart, tstop = roi.start[0], roi.stop[0]
        for t in range(tstart, tstop):

            centers = self.RegionCenters[t].wait()[t]
            centers = numpy.asarray(centers, dtype=numpy.uint32)
            if centers.size:
                centers = centers[1:,:]
            for center in centers:
                x, y, z = center[0:3]
                for dim in (1, 2, 3):
                    for offset in (-1, 0, 1):
                        c = [t, x, y, z, 0]
                        c[dim] += offset
                        c = tuple(c)
                        if self.__contained_in_subregion(roi, c):
                            result[self.__make_key(roi, c)] = 255
        return result

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.RegionCenters:
            self.Output.setDirty([])
