from ilastik.applets.base.standardApplet import StandardApplet

from opObjectClassification import OpObjectClassification
from objectClassificationGui import ObjectClassificationGui
from objectClassificationSerializer import ObjectClassificationSerializer

from lazyflow.graph import OperatorWrapper

class ObjectClassificationApplet( StandardApplet ):
    def __init__(self, 
                 name="Object Classification",                 
                 workflow=None,
                 projectFileGroupName="ObjectClassification"):
        super(ObjectClassificationApplet, self).__init__( name=name, workflow=workflow )
        self._serializableItems = [
            ObjectClassificationSerializer(self.topLevelOperator,
                                           projectFileGroupName)]
        
    @property
    def singleLaneOperatorClass( self ):
        return OpObjectClassification

    @property
    def broadcastingSlots( self ):
        return []

    @property
    def singleLaneGuiClass( self ):
        return ObjectClassificationGui

    @property
    def dataSerializers(self):
        return self._serializableItems
