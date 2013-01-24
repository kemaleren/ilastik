from PyQt4.QtGui import *
from PyQt4 import uic
from PyQt4.QtCore import pyqtSlot

from ilastik.widgets.featureTableWidget import FeatureEntry
from ilastik.widgets.featureDlg import FeatureDlg

import os
import numpy
from ilastik.utility import bind
from lazyflow.operators import OpSubRegion

import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger('TRACE.' + __name__)

from lazyflow.tracer import traceLogged
from ilastik.applets.layerViewer import LayerViewerGui
from ilastik.applets.labeling import LabelingGui

import volumina.colortables as colortables
from volumina.api import \
    LazyflowSource, GrayscaleLayer, ColortableLayer, AlphaModulatedLayer, \
    ClickableColortableLayer, LazyflowSinkSource

from volumina.brushingcontroler import ClickInterpreter2


class ObjectClassificationGui(LabelingGui):

    def centralWidget(self):
        return self

    def appletDrawers(self):
        # Get the labeling drawer from the base class
        labelingDrawer = super(ObjectClassificationGui, self).appletDrawers()[0][1]
        return [("Training", labelingDrawer)]

    def reset(self):
        # Base class first
        super(ObjectClassificationGui, self).reset()

        # Ensure that we are NOT in interactive mode
        self.labelingDrawerUi.checkInteractive.setChecked(False)
        self.labelingDrawerUi.checkShowPredictions.setChecked(False)

    @traceLogged(traceLogger)
    def __init__(self, op, shellRequestSignal, guiControlSignal):
        # Tell our base class which slots to monitor
        labelSlots = LabelingGui.LabelingSlots()
        labelSlots.labelInput = op.LabelInputs
        labelSlots.labelOutput = op.LabelImages

        labelSlots.labelEraserValue = op.Eraser
        labelSlots.labelDelete = op.DeleteLabel

        labelSlots.maxLabelValue = op.NumLabels
        labelSlots.labelsAllowed = op.LabelsAllowedFlags

        # We provide our own UI file (which adds an extra control for interactive mode)
        # This UI file is copied from pixelClassification pipeline
        #
        labelingDrawerUiPath = os.path.split(__file__)[0] + '/labelingDrawer.ui'

        # Base class init
        super(ObjectClassificationGui, self).__init__(labelSlots, op, labelingDrawerUiPath)

        self.op = op
        self.guiControlSignal = guiControlSignal
        self.shellRequestSignal = shellRequestSignal

        self.interactiveModeActive = False

        self.labelingDrawerUi.checkInteractive.setEnabled(True)
        self.labelingDrawerUi.checkInteractive.toggled.connect(self.toggleInteractive)
        self.labelingDrawerUi.checkShowPredictions.setEnabled(True)
        self.labelingDrawerUi.checkShowPredictions.toggled.connect(self.handleShowPredictionsClicked)

        self.op.NumLabels.notifyDirty(bind(self.handleLabelSelectionChange))

    @traceLogged(traceLogger)
    def initAppletDrawerUi(self):
        """
        Load the ui file for the applet drawer, which we own.
        """
        localDir = os.path.split(__file__)[0]
        # (We don't pass self here because we keep the drawer ui in a separate object.)
        self.drawer = uic.loadUi(localDir+"/drawer.ui")

    @traceLogged(traceLogger)
    def createLabelLayer(self, direct=False):
        """Return a colortable layer that displays the label slot
        data, along with its associated label source.

        direct: whether this layer is drawn synchronously by volumina

        """
        labelInput = self._labelingSlots.labelInput
        labelOutput = self._labelingSlots.labelOutput

        if not labelOutput.ready():
            return (None, None)
        else:
            traceLogger.debug("Setting up labels for image index")
            labelsrc = LazyflowSinkSource(labelOutput,
                                          labelInput)
            labellayer = ClickableColortableLayer(self.editor,
                                                  self.onClick,
                                                  datasource=labelsrc,
                                                  colorTable=self._colorTable16,
                                                  direct=direct)

            labellayer.segmentationImageSlot = self.op.SegmentationImagesOut
            labellayer.name = "Labels"
            labellayer.ref_object = None
            labellayer.zeroIsTransparent  = False
            labellayer.colortableIsRandom = True

            # FIXME: labeling actions should not be tied to this
            # particular layer.
            clickInt = ClickInterpreter2(self.editor, labellayer,
                                         self.onClick)
            self.editor.brushingInterpreter = clickInt

            return labellayer, labelsrc

    @traceLogged(traceLogger)
    def setupLayers(self):

        # Base class provides the label layer.
        layers = super(ObjectClassificationGui, self).setupLayers()
        #This is just for colors
        labels = self.labelListData

        labelOutput = self._labelingSlots.labelOutput
        binarySlot = self.op.BinaryImages

        if binarySlot.ready():
            ct = colortables.create_default_8bit()
            self.binaryimagesrc = LazyflowSource(binarySlot)
            layer = GrayscaleLayer(self.binaryimagesrc, range=(0,1), normalize=(0,1))
            layer.name = "Binary Image"
            layers.append(layer)

        predictionSlot = self.op.PredictionImages
        if predictionSlot.ready():
            self.predictsrc = LazyflowSource(predictionSlot)
            self.predictlayer = ColortableLayer(self.predictsrc, colorTable=self._colorTable16)
            self.predictlayer.name = "Prediction"
            self.predictlayer.ref_object = None
            self.predictlayer.visible = self.labelingDrawerUi.checkInteractive.isChecked()

            # put first, so that it is visible after hitting "live
            # predict".
            layers.insert(0, self.predictlayer)

        return layers

    @pyqtSlot()
    @traceLogged(traceLogger)
    def handleLabelSelectionChange(self):
        enabled = False
        if self.op.NumLabels.ready():
            enabled = True
            enabled &= self.op.NumLabels.value >= 2

        self.labelingDrawerUi.savePredictionsButton.setEnabled(enabled)
        self.labelingDrawerUi.checkInteractive.setEnabled(enabled)
        self.labelingDrawerUi.checkShowPredictions.setEnabled(enabled)

    @traceLogged(traceLogger)
    def toggleInteractive(self, checked):
        logger.debug("toggling interactive mode to '%r'" % checked)
        if checked and len(self.op.ObjectFeatures) == 0:
            self.labelingDrawerUi.checkInteractive.setChecked(False)
            mexBox=QMessageBox()
            mexBox.setText("There are no features selected ")
            mexBox.exec_()
            return

        self.labelingDrawerUi.savePredictionsButton.setEnabled(not checked)
        self.op.FreezePredictions.setValue(not checked)

        # Auto-set the "show predictions" state according to what the
        # user just clicked.
        if checked:
            self.labelingDrawerUi.checkShowPredictions.setChecked(True)
            self.handleShowPredictionsClicked()

        # If we're changing modes, enable/disable our controls and
        # other applets accordingly
        if self.interactiveModeActive != checked:
            if checked:
                self.labelingDrawerUi.labelListView.allowDelete = False
                self.labelingDrawerUi.AddLabelButton.setEnabled(False)
            else:
                self.labelingDrawerUi.labelListView.allowDelete = True
                self.labelingDrawerUi.AddLabelButton.setEnabled(True)
        self.interactiveModeActive = checked

    @pyqtSlot()
    @traceLogged(traceLogger)
    def handleShowPredictionsClicked(self):
        checked = self.labelingDrawerUi.checkShowPredictions.isChecked()
        for layer in self.layerstack:
            if "Prediction" in layer.name:
                layer.visible = checked

        # If we're being turned off, turn off live prediction mode, too.
        if not checked and self.labelingDrawerUi.checkInteractive.isChecked():
            self.labelingDrawerUi.checkInteractive.setChecked(False)
            # And hide all segmentation layers
            for layer in self.layerstack:
                if "Segmentation" in layer.name:
                    layer.visible = False

    def onClick(self, layer, pos5D, pos):
        """Extracts the object index that was clicked on and updates
        that object's label.

        """
        label = self.editor.brushingModel.drawnNumber
        slicing = tuple(slice(i, i+1) for i in pos5D)

        # FIXME: this does not work with slot.[slicing].wait()
        arr = layer.segmentationImageSlot.value[slicing]
        obj = arr.flat[0]
        if obj == 0: # background
            return
        labelslot = layer._datasources[0]._inputSlot
        labels = labelslot.value
        labels[pos5D[0]][obj] = label
        labelslot.setValue(labels)
        labelslot.setDirty(slice(None))
