from lazyflow.graph import Graph, Operator, OperatorWrapper

from ilastik.workflow import Workflow

from ilastik.applets.projectMetadata import ProjectMetadataApplet
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.featureSelection import FeatureSelectionApplet
from ilastik.applets.pixelClassification import \
    PixelClassificationApplet, PixelClassificationBatchResultsApplet
from ilastik.applets.objectExtraction import ObjectExtractionApplet
from ilastik.applets.objectClassification import ObjectClassificationApplet

from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.stype import Opaque
from lazyflow.operators.ioOperators.opInputDataReader import OpInputDataReader
from lazyflow.operators import OpAttributeSelector, OpSegmentation, Op5ifyer


class ObjectClassificationWorkflow(Workflow):
    name = "Object Classification Workflow"

    def __init__(self, headless, *args, **kwargs):
        graph = kwargs['graph'] if 'graph' in kwargs else Graph()
        if 'graph' in kwargs: del kwargs['graph']
        super(ObjectClassificationWorkflow, self).__init__(headless=headless, graph=graph, *args, **kwargs)

        ######################
        # Interactive workflow
        ######################

        self.projectMetadataApplet = ProjectMetadataApplet()

        self.dataSelectionApplet = DataSelectionApplet(self,
                                                       "Input: Segmentation",
                                                       "Input Segmentation",
                                                       batchDataGui=False,
                                                       force5d=True)

        self.featureSelectionApplet = FeatureSelectionApplet(self,
                                                             "Feature Selection",
                                                             "FeatureSelections")

        self.pcApplet = PixelClassificationApplet(self, "PixelClassification")
        self.objectExtractionApplet = ObjectExtractionApplet(workflow=self)
        self.objectClassificationApplet = ObjectClassificationApplet(workflow=self)

        self._applets = []
        self._applets.append(self.projectMetadataApplet)
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.featureSelectionApplet)
        self._applets.append(self.pcApplet)
        self._applets.append(self.objectExtractionApplet)
        self._applets.append(self.objectClassificationApplet)


    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName

    def connectLane(self, laneIndex):
        ## Access applet operators
        opData = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opTrainingFeatures = self.featureSelectionApplet.topLevelOperator.getLane(laneIndex)
        opClassify = self.pcApplet.topLevelOperator.getLane(laneIndex)
        opObjExtraction = self.objectExtractionApplet.topLevelOperator.getLane(laneIndex)
        opObjClassification = self.objectClassificationApplet.topLevelOperator.getLane(laneIndex)

        # connect input image
        opTrainingFeatures.InputImage.connect(opData.Image)
        opClassify.InputImages.connect(opData.Image)
        opObjExtraction.RawImage.connect(opData.Image)
        opObjClassification.BinaryImages.connect(opData.Image)

        # training flags
        opClassify.LabelsAllowedFlags.connect(opData.AllowLabels)
        opObjClassification.LabelsAllowedFlags.connect(opData.AllowLabels)

        # connect pixel features
        opClassify.FeatureImages.connect(opTrainingFeatures.OutputImage)
        opClassify.CachedFeatureImages.connect(opTrainingFeatures.CachedOutputImage)

        # connect pixel output to object extraction
        # TODO: how to put operators between applets?
        opseg = OpSegmentation(graph=self.graph)
        op5 = Op5ifyer(graph=self.graph)
        opseg.Input.connect(opClassify.PredictionProbabilities)
        op5.input.connect(opseg.Output)
        opObjExtraction.BinaryImage.connect(op5.output)

        # connect object features
        opObjClassification.SegmentationImages.connect(opObjExtraction.LabelImage)
        opObjClassification.ObjectFeatures.connect(opObjExtraction.RegionFeatures)
