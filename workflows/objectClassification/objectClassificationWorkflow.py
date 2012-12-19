from lazyflow.graph import Graph, Operator, OperatorWrapper

from ilastik.workflow import Workflow

from ilastik.applets.projectMetadata import ProjectMetadataApplet
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.objectExtraction import ObjectExtractionApplet

from ilastik.applets.objectClassification import ObjectClassificationApplet

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.operators.ioOperators.opInputDataReader import OpInputDataReader
from lazyflow.operators import OpAttributeSelector

class ObjectClassificationWorkflow(Workflow):

    def __init__(self):
        graph = Graph()
        super(ObjectClassificationWorkflow, self).__init__(graph = graph)
        self._applets = []

        ######################
        # Interactive workflow
        ######################

        ## Create applets
        self.projectMetadataApplet = ProjectMetadataApplet()
        self.dataSelectionApplet = DataSelectionApplet(self,
                                                       "Input Segmentation",
                                                       "Input Segmentation",
                                                       batchDataGui=False)
        self.objectExtractionApplet = ObjectExtractionApplet(self)
        self.objectClassificationApplet = ObjectClassificationApplet(self)

        ## Access applet operators
        opData = self.dataSelectionApplet.topLevelOperator
        opObjExtraction = self.objectExtractionApplet.topLevelOperator
        opObjClassification = self.objectClassificationApplet.topLevelOperator

        # connect data -> extraction
        opObjExtraction.BinaryImage.connect(opData.Image)

        # connect data -> classification
        opObjClassification.BinaryImages.connect(opData.Image)
        opObjClassification.LabelsAllowedFlags.connect(opData.AllowLabels)

        # connect extraction -> classification
        opObjClassification.SegmentationImages.connect(opObjExtraction.SegmentationImage)
        opObjClassification.ObjectFeatures.connect(opObjExtraction.RegionFeatures)
        opObjClassification.ObjectCounts.connect(opObjExtraction.ObjectCounts)

        self._applets.append(self.projectMetadataApplet)
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.objectExtractionApplet)
        self._applets.append(self.objectClassificationApplet)

        # The shell needs a slot from which he can read the list of
        # image names to switch between. Use an OpAttributeSelector to
        # create a slot containing just the filename from the
        # OpDataSelection's DatasetInfo slot.
        opSelectFilename = OperatorWrapper(OpAttributeSelector, graph=graph)
        opSelectFilename.InputObject.connect(opData.Dataset)
        opSelectFilename.AttributeName.setValue('filePath')

        self._imageNameListSlot = opSelectFilename.Result

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self._imageNameListSlot
