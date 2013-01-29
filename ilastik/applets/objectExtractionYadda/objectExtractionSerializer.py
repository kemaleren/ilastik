from ilastik.applets.base.appletSerializer import AppletSerializer
from lazyflow.rtype import List

class ObjectExtractionSerializer(AppletSerializer):
    def __init__(self, projectFileGroupName, operator):
        slots = []
        super(ObjectExtractionSerializer, self).__init__(
            projectFileGroupName, slots, operator=operator)
