from ilastik.applets.base.applet import Applet
from opPixelClassification import OpPixelClassification
from pixelClassificationSerializer import PixelClassificationSerializer, Ilastik05ImportDeserializer

class PixelClassificationApplet( Applet ):
    """
    Implements the pixel classification "applet", which allows the ilastik shell to use it.
    """
    def __init__( self, workflow, projectFileGroupName ):
        Applet.__init__( self, "Pixel Classification" )

        self._topLevelOperator = OpPixelClassification( parent=workflow )

        # We provide two independent serializing objects:
        #  one for the current scheme and one for importing old projects.
        self._serializableItems = [PixelClassificationSerializer(self._topLevelOperator, projectFileGroupName), # Default serializer for new projects
                                   Ilastik05ImportDeserializer(self._topLevelOperator)]   # Legacy (v0.5) importer


        self._gui = None
        
        # GUI needs access to the serializer to enable/disable prediction storage
        self.predictionSerializer = self._serializableItems[0]

        # FIXME: For now, we can directly connect the progress signal from the classifier training operator
        #  directly to the applet's overall progress signal, because it's the only thing we report progress for at the moment.
        # If we start reporting progress for multiple tasks that might occur simulatneously,
        #  we'll need to aggregate the progress updates.
        self._topLevelOperator.opTrain.progressSignal.subscribe(self.progressSignal.emit)
    
    @property
    def topLevelOperator(self):
        return self._topLevelOperator

    @property
    def dataSerializers(self):
        return self._serializableItems

    @property
    def gui(self):
        if self._gui is None:
            from pixelClassificationGui import PixelClassificationGui
            self._gui = PixelClassificationGui( self._topLevelOperator, self.shellRequestSignal, self.guiControlSignal, self.predictionSerializer )        
        return self._gui
