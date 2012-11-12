from ilastik.applets.base.applet import Applet

from opObjectClassification import OpObjectClassification
from objectClassificationGui import ObjectClassificationGui

from lazyflow.graph import OperatorWrapper

class ObjectClassificationApplet(Applet):
    def __init__(self, workflow,
                 guiName="Object Features",
                 projectFileGroupName="ObjectFeatures"):
        super(ObjectClassificationApplet, self).__init__(guiName)
        self._topLevelOperator = OpObjectClassification(parent=workflow)
        self._gui = ObjectClassificationGui(self._topLevelOperator,
                                            self.guiControlSignal,
                                            self.shellRequestSignal)
        self._serializableItems = []
        
    @property
    def topLevelOperator(self):
        return self._topLevelOperator

    @property
    def dataSerializers(self):
        return self._serializableItems

    @property
    def viewerControlWidget(self):
        return self._centralWidget.viewerControlWidget

    @property
    def gui(self):
        return self._gui