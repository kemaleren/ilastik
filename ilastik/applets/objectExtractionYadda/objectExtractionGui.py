from PyQt4.QtGui import QWidget, QColor, QVBoxLayout, QProgressDialog
from PyQt4 import uic
from PyQt4.QtCore import Qt, QString

from lazyflow.rtype import SubRegion

import os

from volumina.api import \
    LazyflowSource, GrayscaleLayer, RGBALayer, \
    ConstantSource, AlphaModulatedLayer, LayerStackModel, \
    VolumeEditor, VolumeEditorWidget, ColortableLayer
import volumina.colortables as colortables

from ilastik.applets.layerViewer import LayerViewerGui

import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger('TRACE.' + __name__)
from lazyflow.tracer import Tracer

import vigra

class ObjectExtractionGui( LayerViewerGui ):
    """
    """

    ###########################################
    ### AppletGuiInterface Concrete Methods ###
    ###########################################
    def centralWidget( self ):
        return self.volumeEditorWidget

    def appletDrawers( self ):
        return [ ("Object Extraction", self._drawer ) ]

    def menus( self ):
        return []

    def viewerControlWidget( self ):
        return self._viewerControlWidget

    def setupLayers(self):

        layers = []
        mainOperator = self.mainOperator

        binarySlot = mainOperator.BinaryImage
        segmentationSlot = mainOperator.SegmentationImage
        centerSlot = mainOperator.ObjectCenterImage


        if binarySlot.ready():
            ct = colortables.create_default_8bit()
            self.binaryimagesrc = LazyflowSource( binarySlot )
            layer = GrayscaleLayer( self.binaryimagesrc, range=(0,1), normalize=(0,1) )
            layer.name = "Binary Image"
            layers.append(layer)
            #self.layerstack.append(layer)
        if segmentationSlot.ready():
            ct = colortables.create_default_16bit()
            self.objectssrc = LazyflowSource( segmentationSlot )
            ct[0] = QColor(0,0,0,0).rgba() # make 0 transparent
            layer = ColortableLayer( self.objectssrc, ct )
            layer.name = "Label Image"
            layer.opacity = 0.5
            #self.layerstack.append(layer)
            layers.append(layer)

        if centerSlot.ready():
            self.centerimagesrc = LazyflowSource( centerSlot )
            layer = RGBALayer( red=ConstantSource(255), alpha=self.centerimagesrc )
            layer.name = "Object Centers"
            layers.append(layer)
            #self.layerstack.append( layer )
        return layers

    def reset( self ):
        pass

    ###########################################
    ###########################################

    def __init__(self, mainOperator):
        """
        """
        super(ObjectExtractionGui, self).__init__(mainOperator)
        self.mainOperator = mainOperator

    def initAppletDrawerUi(self):
        # Load the ui file (find it in our own directory)
        localDir = os.path.split(__file__)[0]
        self._drawer = uic.loadUi(localDir+"/drawer.ui")

#        self._drawer.labelImageButton.pressed.connect(self._onSegmentationImageButtonPressed)
        self._drawer.extractObjectsButton.pressed.connect(self._onExtractObjectsButtonPressed)

    def initViewerControlUi( self ):
        p = os.path.split(__file__)[0]+'/'
        if p == "/": p = "."+p
        pdir = p.split("/")[1:-2]
        path = ""
        for pd in pdir:
            path = path+"/"+pd
        path = path + "/featureSelection/viewerControls.ui"
        self._viewerControlWidget = uic.loadUi(path)
        layerListWidget = self._viewerControlWidget.featureListWidget

        # Need to handle data changes because the layerstack model hasn't
        # updated his data yet by the time he calls the rowsInserted signal
        def handleLayerStackDataChanged(startIndex, stopIndex):
            row = startIndex.row()
            layerListWidget.item(row).setText(self.layerstack[row].name)
        self.layerstack.dataChanged.connect(handleLayerStackDataChanged)

        def handleInsertedLayers(parent, start, end):
            for i in range(start, end+1):
                layerListWidget.insertItem(i, self.layerstack[i].name)
        self.layerstack.rowsInserted.connect( handleInsertedLayers )

        def handleRemovedLayers(parent, start, end):
            for i in reversed(range(start, end+1)):
                layerListWidget.takeItem(i)
        self.layerstack.rowsRemoved.connect( handleRemovedLayers )

        def handleSelectionChanged(row):
            # Only one layer is visible at a time
            for i, layer in enumerate(self.layerstack):
                layer.visible = (i == row)
        layerListWidget.currentRowChanged.connect( handleSelectionChanged )


    def _onExtractObjectsButtonPressed( self ):

        oper = self.mainOperator
        m = oper.SegmentationImage.meta
        if m.axistags.axisTypeCount(vigra.AxisType.Time) >0:
            maxt = oper.SegmentationImage.meta.shape[0]
            progress = QProgressDialog("Extracting objects...", "Stop", 0, maxt)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.setCancelButtonText(QString())

            reqs = []
            oper._opRegFeats.fixed = False
            for t in range(maxt):
                reqs.append(oper.RegionFeatures([t]))
                reqs[-1].submit()
            for i, req in enumerate(reqs):
                progress.setValue(i)
                if progress.wasCanceled():
                    req.cancel()
                else:
                    req.wait()

            oper._opRegFeats.fixed = True
            progress.setValue(maxt)
        else:
            oper._opRegFeats.fixed = False
            oper.RegionFeatures([0]).wait()
            oper._opRegFeats.fixed = True

        oper.ObjectCenterImage.setDirty( SubRegion(oper.ObjectCenterImage))
