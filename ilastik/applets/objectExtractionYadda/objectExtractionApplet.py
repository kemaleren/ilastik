from ilastik.applets.base.standardApplet import StandardApplet

from opObjectExtraction import OpObjectExtraction
from objectExtractionGui import ObjectExtractionGui
from objectExtractionSerializer import ObjectExtractionSerializer

class ObjectExtractionApplet(StandardApplet):
    def __init__(self, name="Object Extraction", workflow=None,
                 projectFileGroupName="ObjectExtraction"):
        super(ObjectExtractionApplet, self).__init__(name=name,
                                                     workflow=workflow)
        self._serializableItems = [
            ObjectExtractionSerializer(projectFileGroupName,
                                       self.topLevelOperator)]


    @property
    def singleLaneOperatorClass(self):
        return OpObjectExtraction

    @property
    def broadcastingSlots(self):
        return []

    @property
    def singleLaneGuiClass(self):
        return ObjectExtractionGui

    @property
    def dataSerializers(self):
        return self._serializableItems
