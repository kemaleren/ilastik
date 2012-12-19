from ilastik.applets.base.appletSerializer import AppletSerializer
from lazyflow.rtype import List

class ObjectExtractionSerializer(AppletSerializer):
    """
    """
    SerializerVersion = 0.1
    _featureNames = ['RegionCenter', 'Count']

    def __init__(self, mainOperator, projectFileGroupName):
        super(ObjectExtractionSerializer, self).__init__(
            projectFileGroupName, self.SerializerVersion)
        self.mainOperator = mainOperator

    def _serializeToHdf5(self, topGroup, hdf5File, projectFilePath):
        for i, op in enumerate(self.mainOperator.innerOperators):
            opname = "operator_{0}".format(i)
            self.deleteIfPresent(topGroup, opname)
            group = self.getOrCreateGroup(topGroup, opname)
            self._serializeOperatorToHdf5(op, group, hdf5File,
                                          projectFilePath)

    def _deserializeFromHdf5(self, topGroup, groupVersion, hdf5File,
                             projectFilePath):
        for i, op in enumerate(self.mainOperator.innerOperators):
            opname = "operator_{0}".format(i)
            if opname in topGroup.keys():
                group = topGroup[opname]
                self._deserializeOperatorFromHdf5(op, group,
                                                  groupVersion,
                                                  hdf5File,
                                                  projectFilePath)


    def _serializeOperatorToHdf5(self, op, topGroup, hdf5File,
                                 projectFilePath):
        self.deleteIfPresent(topGroup, "SegmentationImage")
        topGroup.create_dataset('SegmentationImage', data=op.SegmentationImage.value)

        self.deleteIfPresent(topGroup, "samples")
        samples_gr = self.getOrCreateGroup(topGroup, "samples")
        for t in op._opRegFeats._cache.keys():
            t_gr = samples_gr.create_group(str(t))
            for name in self._featureNames:
                t_gr.create_dataset(name=name,
                                    data=op._opRegFeats._cache[t][name])

    def _deserializeOperatorFromHdf5(self, op, topGroup, groupVersion,
                                     hdf5File, projectFilePath):
        top.SegmentationImage.setValue(topGroup['SegmentationImage'][()])

        if "samples" in topGroup.keys():
            cache = {}
            for t in topGroup["samples"].keys():
                cache[int(t)] = dict()
                for name in self._featureNames:
                    if name in topGroup["samples"][t].keys():
                        cache[int(t)][name] = topGroup["samples"][t][name].value
            op._opRegFeats._cache = cache

        # update region count
        slot = op.RegionCount
        roi = List(slot, [0])
        op._regionCount(roi)

    def isDirty(self):
        return True

    def unload(self):
        pass
