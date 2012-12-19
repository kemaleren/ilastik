from ilastik.applets.base.appletSerializer import AppletSerializer


class Groups(object):
    Labels = "Labels"


class ObjectClassificationSerializer(AppletSerializer):
    """
    """
    SerializerVersion = 0.1

    def __init__(self, mainOperator, projectFileGroupName):
        super(ObjectClassificationSerializer, self ).__init__(projectFileGroupName, self.SerializerVersion)
        self.mainOperator = mainOperator

    def _serializeToHdf5(self, topGroup, hdf5File, projectFilePath):
        op = self.mainOperator
        self.deleteIfPresent(topGroup, Groups.Labels)
        labels_gr = self.getOrCreateGroup(topGroup, Groups.Labels)
        for i in range(len(op.LabelInputs)):
            name = str(i)
            labels_gr.create_dataset(name=name, data=op.LabelInputs[i].value[0])


    def _deserializeFromHdf5(self, topGroup, groupVersion, hdf5File, projectFilePath):
        op = self.mainOperator
        if Groups.Labels in topGroup.keys():
            for key in topGroup[Groups.Labels].keys():
                i = int(key)
                value = topGroup[Groups.Labels][key].value
                op.LabelInputs[i].setValue([value])


    def isDirty(self):
        return True

    def unload(self):
        pass
