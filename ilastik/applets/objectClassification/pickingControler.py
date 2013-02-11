from PyQt4.QtCore import pyqtSignal, QObject, QEvent, QPointF, Qt, QRectF
from PyQt4.QtGui import QPainter, QPen, QBrush, QApplication, QGraphicsView

from volumina.eventswitch import InterpreterABC
from volumina.navigationControler import NavigationInterpreter
from volumina.brushingcontroler import BrushingInterpreter

class PickingInterpreter( BrushingInterpreter ):
    # states
    FINAL = 0
    DEFAULT_MODE = 1
    DRAW_MODE = 2

    @property
    def state( self ):
        return self._current_state

    def __init__( self, navigationControler, pickingControler ):
        QObject.__init__( self )
        self._navCtrl = navigationControler
        self._navIntr = NavigationInterpreter( navigationControler )
        self._pickingCtrl = pickingControler
        self._current_state = self.FINAL
        self._temp_erasing = False # indicates, if user pressed shift
                                   # for temporary erasing (in
                                   # contrast to selecting the eraser brush)

        self._lineItems = [] # list of line items that have been
                            # added to the qgraphicsscene for drawing indication

        # clear the temporary line items once they
        # have been pushed to the sink
        self._pickingCtrl.wroteToSink.connect(self.clearLines)

    def start( self ):
        if self._current_state == self.FINAL:
            self._navIntr.start()
            self._current_state = self.DEFAULT_MODE
        else:
            pass # ignore

    def stop( self ):
        if self._pickingCtrl._isDrawing:
            for imageview in self._navCtrl._views:
                self._pickingCtrl.endDrawing(imageview.mousePos)
        self._current_state = self.FINAL
        self._navIntr.stop()

    def eventFilter( self, watched, event ):
        etype = event.type()
        ### the following implements a simple state machine
        if self._current_state == self.DEFAULT_MODE:
            ### default mode -> draw mode
            if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and event.modifiers() == Qt.NoModifier:
                # navigation interpreter also has to be in 
                # default mode to avoid inconsistencies
                if self._navIntr.state == self._navIntr.DEFAULT_MODE:
                    self._current_state = self.DRAW_MODE
                    self.onEntry_draw( watched, event )
                    return True
                else:
                    return self._navIntr.eventFilter( watched, event )

            ### actions in default mode
            # let the navigation interpreter handle common events
            return self._navIntr.eventFilter( watched, event )

        elif self._current_state == self.DRAW_MODE:
            ### draw mode -> default mode
            if etype == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.onExit_draw( watched, event )
                self._current_state = self.DEFAULT_MODE
                self.onEntry_default( watched, event )
                return True
            
            ### actions in draw mode
            elif etype == QEvent.MouseMove:
                self.onMouseMove_draw( watched, event )
                return True

        return False

    ###
    ### Default Mode
    ###
    def onEntry_default( self, imageview, event ):
        pass

    ###
    ### Draw Mode
    ###
    def onEntry_draw( self, imageview, event ):
        if QApplication.keyboardModifiers() == Qt.ShiftModifier:
            self._pickingCtrl._pickingModel.setErasing()
            self._temp_erasing = True
        imageview.mousePos = imageview.mapScene2Data(imageview.mapToScene(event.pos()))
        self._pickingCtrl.beginDrawing(imageview, imageview.mousePos)
    
    def onExit_draw( self, imageview, event ):
        self._pickingCtrl.endDrawing(imageview.mousePos)
        if self._temp_erasing:
            self._pickingCtrl._pickingModel.disableErasing()
            self._temp_erasing = False

    def onMouseMove_draw( self, imageview, event ):
        self._navIntr.onMouseMove_default( imageview, event )
        '''
        o = imageview.scene().data2scene.map(QPointF(imageview.oldX,imageview.oldY))
        n = imageview.scene().data2scene.map(QPointF(imageview.x,imageview.y))
        pen = QPen( QBrush(self._brushingCtrl._brushingModel.drawColor), self._brushingCtrl._brushingModel.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        line = imageview.scene().addLine(o.x(), o.y(), n.x(), n.y(), pen)
        self._lineItems.append(line)
        
        self._pickingCtrl._pickingModel.moveTo(imageview.mousePos)
        '''
        
    def clearLines(self):
        lines = self._lineItems
        self._lineItems = []
        for l in lines:
            l.hide()
        
#*******************************************************************************
# B r u s h i n g C o n t r o l e r                                            *
#*******************************************************************************

class PickingControler(QObject):
    wroteToSink     = pyqtSignal()

    def __init__(self, pickingModel, positionModel, dataSink):
        QObject.__init__(self, parent=None)
        self._dataSink = dataSink
        
        self._pickingModel = pickingModel
        self._pickingModel.brushStrokeAvailable.connect(self._writeIntoSink)
        self._positionModel = positionModel

        self._isDrawing = False
        self._tempErase = False

    def beginDrawing(self, imageview, pos):
        imageview.mousePos = pos
        self._isDrawing  = True
        self._pickingModel.beginDrawing(pos, imageview.sliceShape)

    def endDrawing(self, pos): 
        self._isDrawing = False
        self._pickingModel.endDrawing(pos)
        
    def setDataSink(self, dataSink):
        self._dataSink = dataSink
        
    def _writeIntoSink(self, brushStrokeOffset, labels):
        activeView = self._positionModel.activeView
        slicingPos = self._positionModel.slicingPos
        t, c       = self._positionModel.time, self._positionModel.channel
        
        slicing = [slice(brushStrokeOffset.x(), brushStrokeOffset.x()+labels.shape[0]), \
                   slice(brushStrokeOffset.y(), brushStrokeOffset.y()+labels.shape[1])]
        slicing.insert(activeView, slice(slicingPos[activeView], slicingPos[activeView]+1))
        slicing = (slice(t,t+1),) + tuple(slicing) + (slice(c,c+1),)
        
        #make the labels 5d for correct graph compatibility
        newshape = list(labels.shape)
        newshape.insert(activeView, 1)
        newshape.insert(0, 1)
        newshape.append(1)
        
        #newlabels = numpy.zeros
        
        self._dataSink.put(slicing, labels.reshape(tuple(newshape)))
        self.wroteToSink.emit()


