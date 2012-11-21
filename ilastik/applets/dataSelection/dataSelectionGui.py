from PyQt4.QtCore import Qt, QVariant
from PyQt4.QtGui import *
from PyQt4 import uic

from opDataSelection import OpDataSelection, DatasetInfo

from functools import partial
import os
import copy
import glob
import threading
import h5py

from volumina.utility import PreferencesManager

from ilastik.shell.gui.iconMgr import ilastikIcons
from ilastik.utility import bind
from ilastik.utility.gui import ThreadRouter, threadRouted
from ilastik.utility.pathHelpers import getPathVariants

from ilastik.applets.base.applet import ControlCommand

import vigra

import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger('TRACE.' + __name__)
from lazyflow.tracer import Tracer

class Column():
    """ Enum for table column positions """
    Name = 0
    Location = 1
    InternalID = 2
    LabelsAllowed = 3 # Note: For now, this column must come last because it gets removed in batch mode.
    
    NumColumns = 4

class LocationOptions():
    """ Enum for location menu options """
    Project = 0
    AbsolutePath = 1
    RelativePath = 2

class GuiMode():
    Normal = 0
    Batch = 1

class DataSelectionGui(QMainWindow):
    """
    Manages all GUI elements in the data selection applet.
    This class itself is the central widget and also owns/manages the applet drawer widgets.
    """
    
    ###########################################
    ### AppletGuiInterface Concrete Methods ###
    ###########################################

    def centralWidget( self ):
        return self

    def appletDrawers(self):
        return [ (self.title, self.drawer) ]
    
    def menus( self ):
        return []

    def viewerControlWidget(self):
        return None # No viewer controls for this applet.
    
    def setImageIndex(self, imageIndex):
        pass # This applet doesn't care which image is currently selected.  It always lists all inputs.

    def reset(self):
        # Nothing to do (our drawer has no state and the central widget is dynamically loaded/unloaded by operator changes)
        pass
        
    ###########################################
    ###########################################

    def __init__(self, dataSelectionOperator, serializer, guiControlSignal, guiMode=GuiMode.Normal, title="Input Selection"):
        with Tracer(traceLogger):
            super(DataSelectionGui, self).__init__()
            
            self.title = title
    
            self.drawer = None
            self.mainOperator = dataSelectionOperator
            self.guiMode = guiMode
            self.serializer = serializer
            self.guiControlSignal = guiControlSignal
            self.threadRouter = ThreadRouter(self)
            
            self.initAppletDrawerUic()
            self.initCentralUic()
    
            def handleNewDataset( multislot, index ):
                with Tracer(traceLogger):
                    assert multislot == self.mainOperator.Dataset
                    # Make room in the table
                    self.fileInfoTableWidget.insertRow( index )
                    
                    # Update the table row data when this slot has new data
                    # We can't bind in the row here because the row may change in the meantime.
                    self.mainOperator.Dataset[index].notifyDirty( self.updateTableForSlot )
    
            self.mainOperator.Dataset.notifyInserted( bind( handleNewDataset ) )
        
            def handleDatasetRemoved( multislot, index ):
                with Tracer(traceLogger):
                    assert multislot == self.mainOperator.Dataset
                    
                    # Simply remove the row we don't need any more
                    self.fileInfoTableWidget.removeRow( index )
    
            self.mainOperator.Dataset.notifyRemove( bind( handleDatasetRemoved ) )
        
    def initAppletDrawerUic(self):
        """
        Load the ui file for the applet drawer, which we own.
        """
        with Tracer(traceLogger):
            # Load the ui file (find it in our own directory)
            localDir = os.path.split(__file__)[0]+'/'
            # (We don't pass self here because we keep the drawer ui in a separate object.)
            self.drawer = uic.loadUi(localDir+"/dataSelectionDrawer.ui")
    
            # Set up our handlers
            self.drawer.addFileButton.clicked.connect(self.handleAddFileButtonClicked)
            self.drawer.addFileButton.setIcon( QIcon(ilastikIcons.AddSel) )

            self.drawer.addStackButton.clicked.connect(self.handleAddStackButtonClicked)
            self.drawer.addStackButton.setIcon( QIcon(ilastikIcons.AddSel) )

            self.drawer.addStackFilesButton.clicked.connect(self.handleAddStackFilesButtonClicked)
            self.drawer.addStackFilesButton.setIcon( QIcon(ilastikIcons.AddSel) )

            self.drawer.removeFileButton.clicked.connect(self.handleRemoveButtonClicked)
            self.drawer.removeFileButton.setIcon( QIcon(ilastikIcons.RemSel) )
    
    def initCentralUic(self):
        """
        Load the GUI from the ui file into this class and connect it with event handlers.
        """
        with Tracer(traceLogger):
            # Load the ui file into this class (find it in our own directory)
            localDir = os.path.split(__file__)[0]+'/'
            uic.loadUi(localDir+"/dataSelection.ui", self)
    
            self.fileInfoTableWidget.resizeRowsToContents()
            self.fileInfoTableWidget.resizeColumnsToContents()
            self.fileInfoTableWidget.setAlternatingRowColors(True)
            self.fileInfoTableWidget.setShowGrid(False)
            self.fileInfoTableWidget.horizontalHeader().setResizeMode(Column.Name, QHeaderView.Interactive)
            self.fileInfoTableWidget.horizontalHeader().setResizeMode(Column.Location, QHeaderView.Interactive)
            self.fileInfoTableWidget.horizontalHeader().setResizeMode(Column.InternalID, QHeaderView.Interactive)
            
            self.fileInfoTableWidget.horizontalHeader().resizeSection(Column.Name, 200)
            self.fileInfoTableWidget.horizontalHeader().resizeSection(Column.Location, 300)
            self.fileInfoTableWidget.horizontalHeader().resizeSection(Column.InternalID, 200)
    
            if self.guiMode == GuiMode.Batch:
                # It doesn't make sense to provide a labeling option in batch mode
                self.fileInfoTableWidget.removeColumn( Column.LabelsAllowed )
                self.fileInfoTableWidget.horizontalHeader().resizeSection(Column.LabelsAllowed, 150)
                self.fileInfoTableWidget.horizontalHeader().setResizeMode(Column.LabelsAllowed, QHeaderView.Fixed)
            
            self.fileInfoTableWidget.verticalHeader().hide()
    
            # Set up handlers
            self.fileInfoTableWidget.itemSelectionChanged.connect(self.handleTableSelectionChange)
        
    def handleAddFileButtonClicked(self):
        """
        The user clicked the "Add File" button.
        Ask him to choose a file (or several) and add them to both 
          the GUI table and the top-level operator inputs.
        """
        # Find the directory of the most recently opened image file
        mostRecentImageFile = PreferencesManager().get( 'DataSelection', 'recent image' )
        if mostRecentImageFile is not None:
            defaultDirectory = os.path.split(mostRecentImageFile)[0]
        else:
            defaultDirectory = os.path.expanduser('~')

        # Launch the "Open File" dialog
        fileNames = self.getImageFileNamesToOpen(defaultDirectory)

        # If the user didn't cancel        
        if len(fileNames) > 0:
            PreferencesManager().set('DataSelection', 'recent image', fileNames[0])
            self.addFileNames(fileNames)
    
    def handleAddStackButtonClicked(self):
        """
        The user clicked the "Import Stack Directory" button.
        """
        # Find the directory of the most recently opened image file
        mostRecentStackDirectory = PreferencesManager().get( 'DataSelection', 'recent stack directory' )
        if mostRecentStackDirectory is not None:
            defaultDirectory = os.path.split(mostRecentStackDirectory)[0]
        else:
            defaultDirectory = os.path.expanduser('~')

        # Launch the "Open File" dialog
        directoryName = QFileDialog.getExistingDirectory(self,
                                                         "Image Stack Directory",
                                                         defaultDirectory,
                                                         options=QFileDialog.Options(QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly))

        # If the user didn't cancel        
        if not directoryName.isNull():
            PreferencesManager().set('DataSelection', 'recent stack directory', str(directoryName))
            globString = self.getGlobString( str(directoryName) )                
            if globString is not None:
                self.importStackFromGlobString( globString )

    def handleAddStackFilesButtonClicked(self):
        """
        The user clicked the "Import Stack Files" button.
        """
        # Find the directory of the most recently opened image file
        mostRecentStackImageFile = PreferencesManager().get( 'DataSelection', 'recent stack image' )
        if mostRecentStackImageFile is not None:
            defaultDirectory = os.path.split(mostRecentStackImageFile)[0]
        else:
            defaultDirectory = os.path.expanduser('~')

        # Launch the "Open File" dialog
        fileNames = self.getImageFileNamesToOpen(defaultDirectory)

        # If the user didn't cancel        
        if len(fileNames) > 0:
            PreferencesManager().set('DataSelection', 'recent stack image', fileNames[0])
            # Convert into one big string, which is accepted by the stack loading operator
            bigString = "//".join( fileNames )
            self.importStackFromGlobString(bigString)

    def getImageFileNamesToOpen(self, defaultDirectory):
        """
        Launch an "Open File" dialog to ask the user for one or more image files.
        """
        extensions = OpDataSelection.SupportedExtensions
        filter = "Image files (" + ' '.join('*.' + x for x in extensions) + ')'
        dlg = QFileDialog( self, "Select Images", defaultDirectory, filter )
        dlg.setOption( QFileDialog.HideNameFilterDetails, False )
        dlg.setOption( QFileDialog.DontUseNativeDialog, False )
        dlg.setViewMode( QFileDialog.Detail )
        dlg.setFileMode( QFileDialog.ExistingFiles )
        
        if dlg.exec_():
            fileNames = dlg.selectedFiles()
        else:
            fileNames = []        

        # Convert from QtString to python str
        fileNames = [str(s) for s in fileNames]
        return fileNames

    def importStackFromGlobString(self, globString):
        """
        The word 'glob' is used loosely here.  See the OpStackLoader operator for details.
        """
        info = DatasetInfo()
        info.filePath = globString
        
        # Allow labels by default if this gui isn't being used for batch data.
        info.allowLabels = ( self.guiMode == GuiMode.Normal )
        
        def importStack():
            self.guiControlSignal.emit( ControlCommand.DisableAll )
            # Serializer will update the operator for us, which will propagate to the GUI.
            self.serializer.importStackAsLocalDataset( info )
            self.guiControlSignal.emit( ControlCommand.Pop )

        importThread = threading.Thread( target=importStack )
        importThread.start()


    def getGlobString(self, directory):
        exts = vigra.impex.listExtensions().split()
        for ext in exts:
            fullGlob = directory + '/*.' + ext
            filenames = glob.glob(fullGlob)
            if len(filenames) > 0:
                # Be helpful: find the longest globstring we can
                prefix = os.path.commonprefix(filenames)
                return prefix + '*.' + ext
        
        # Couldn't find an image file in the directory...
        return None
        

    def addFileNames(self, fileNames):
        """
        Add the given filenames to both the GUI table and the top-level operator inputs.
        """
        with Tracer(traceLogger):
            # Allocate additional subslots in the operator inputs.
            oldNumFiles = len(self.mainOperator.Dataset)
            self.mainOperator.Dataset.resize( oldNumFiles+len(fileNames) )
    
            # Assign values to the new inputs we just allocated.
            # The GUI will be updated by callbacks that are listening to slot changes
            for i, filePath in enumerate(fileNames):
                datasetInfo = DatasetInfo()
                cwd = self.mainOperator.WorkingDirectory.value
                absPath, relPath = getPathVariants(filePath, cwd)

                # Relative by default, unless the file is in a totally different tree from the working directory.
                if len(os.path.commonprefix([cwd, absPath])) > 1: 
                   datasetInfo.filePath = relPath
                else:
                   datasetInfo.filePath = absPath

                h5Exts = ['.ilp', '.h5', '.hdf5']
                if os.path.splitext(datasetInfo.filePath)[1] in h5Exts:
                    datasetNames = self.getPossibleInternalPaths( absPath )
                    if len(datasetNames) > 0:
                        datasetInfo.filePath += str(datasetNames[0])
                    else:
                        raise RuntimeError("HDF5 file has no image datasets")
                
                # Allow labels by default if this gui isn't being used for batch data.
                datasetInfo.allowLabels = ( self.guiMode == GuiMode.Normal )

                self.mainOperator.Dataset[i+oldNumFiles].setValue( datasetInfo )

    @threadRouted
    def updateTableForSlot(self, slot, *args):
        """
        Update the given rows using the top-level operator parameters
        """
        with Tracer(traceLogger):
            
            # Don't update anything if the slot doesn't have data yet
            if not slot.connected():
                return
            
            # Which index is this slot?
            row = -1
            for i in range( len(self.mainOperator.Dataset) ):
                if slot == self.mainOperator.Dataset[i]:
                    row = i
                    break
    
            assert row != -1, "Unknown input slot!"
            
            totalPath = self.mainOperator.Dataset[row].value.filePath
            lastDotIndex = totalPath.rfind('.')
            extensionAndInternal = totalPath[lastDotIndex:]
            extension = extensionAndInternal.split('/')[0]
            externalPath = totalPath[:lastDotIndex] + extension
    
            internalPath = ''
            internalStart = extensionAndInternal.find('/')
            if internalStart != -1:
                internalPath = extensionAndInternal[internalStart:]
    
            fileName = os.path.split(externalPath)[1]
    
            tableWidget = self.fileInfoTableWidget
    
            # Show the filename in the table (defaults to edit widget)
            tableWidget.setItem( row, Column.Name, QTableWidgetItem(fileName) )
            
            # Create and add the combobox for the internal path selection
            self.updateInternalPathComboBox( row, externalPath, internalPath )
    #        tableWidget.setItem( row, Column.InternalID, QTableWidgetItem(internalPath) )
    
            # Subscribe to changes        
            tableWidget.itemChanged.connect( self.handleRowDataChange )
            
            # Create and add the combobox for storage location options
            self.updateStorageOptionComboBox(row, externalPath)
            
            if self.guiMode != GuiMode.Batch:
                # Create and add the checkbox for the 'allow labels' option
                allowLabelsCheckbox = QCheckBox()
                allowLabelsCheckbox.setChecked( self.mainOperator.Dataset[row].value.allowLabels )
                tableWidget.setCellWidget( row, Column.LabelsAllowed, allowLabelsCheckbox )
                allowLabelsCheckbox.stateChanged.connect( partial(self.handleAllowLabelsCheckbox, self.mainOperator.Dataset[row]) )
                
            # Update the operator, in case we need to select a new internal path based on the updated combo options
            # (Won't have any effect if nothing changed this time around.)
            self.updateFilePath(row)
    
    def handleAllowLabelsCheckbox(self, slot, checked):
        """
        The user (un)checked the "allow labels" checkbox in one of the table rows.
        Update the corresponding dataset info in the operator (which is given in the parameter 'slot')
        """
        with Tracer(traceLogger):
            # COPY the dataset so we trigger the slot to be dirty
            newDatasetInfo = copy.copy(slot.value)
            newDatasetInfo.allowLabels = ( checked == Qt.Checked )
            
            # Only update if necessary
            if newDatasetInfo != slot.value:
                slot.setValue( newDatasetInfo )
    
    def updateStorageOptionComboBox(self, row, filePath):
        """
        Create and add the combobox for storage location options
        """
        with Tracer(traceLogger):
            # Determine the relative path to this file
            absPath, relPath = getPathVariants(filePath, self.mainOperator.WorkingDirectory.value)
            # Add a prefixes to make the options clear
            absPath = "Absolute Link: " + absPath
            relPath = "Relative Link: <project directory>/" + relPath
            
            combo = QComboBox()
            options = {} # combo data -> combo text
            options[ LocationOptions.AbsolutePath ] = absPath
            options[ LocationOptions.RelativePath ] = relPath
            
            options[ LocationOptions.Project ] = "Store in Project File"
                    
            for option, text in sorted(options.items()):
                # Add to the combo, storing the option as the item data
                combo.addItem(text, option)
    
            # Select the combo index that matches the current setting
            location = self.mainOperator.Dataset[row].value.location
    
            if location == DatasetInfo.Location.ProjectInternal:
                comboData = LocationOptions.Project
            elif location == DatasetInfo.Location.FileSystem:
                # Determine if the path is relative or absolute
                if os.path.isabs(self.mainOperator.Dataset[row].value.filePath[0]):
                    comboData = LocationOptions.AbsolutePath
                else:
                    comboData = LocationOptions.RelativePath
    
            comboIndex = combo.findData( QVariant(comboData) )
            combo.setCurrentIndex( comboIndex )
    
            combo.currentIndexChanged.connect( partial(self.handleComboSelectionChanged, combo) )
            self.fileInfoTableWidget.setCellWidget( row, Column.Location, combo )
    
    def updateInternalPathComboBox( self, row, externalPath, internalPath ):
        with Tracer(traceLogger):
            combo = QComboBox()
            datasetNames = []
    
            # Make sure we're dealing with the absolute path (to make this simple)
            absPath, relPath = getPathVariants(externalPath, self.mainOperator.WorkingDirectory.value)
            ext = os.path.splitext(absPath)[1]
            h5Exts = ['.ilp', '.h5', '.hdf5']
            if ext in h5Exts:
                datasetNames = self.getPossibleInternalPaths(absPath)

            # Add each dataset option to the combo            
            for path in datasetNames:
                combo.addItem( path )
    
            # If the internal path we used previously is in the combo list, select it.
            prevSelection = combo.findText( internalPath )
            if prevSelection != -1:
                combo.setCurrentIndex( prevSelection )
    
            # Define response to changes and add it to the GUI.
            # Pass in the corresponding the table item so we can figure out which row this came from
            combo.currentIndexChanged.connect( bind(self.handleComboSelectionChanged, combo) )
            self.fileInfoTableWidget.setCellWidget( row, Column.InternalID, combo )
            
            # Since we just selected a new internal path, call the handler 
            #self.handleComboSelectionChanged(combo, combo.currentIndex())

    def getPossibleInternalPaths(self, absPath):
        datasetNames = []
        # Open the file as a read-only so we can get a list of the internal paths
        with h5py.File(absPath, 'r') as f:
            # Define a closure to collect all of the dataset names in the file.
            def accumulateDatasetPaths(name, val):
                if type(val) == h5py._hl.dataset.Dataset and 3 <= len(val.shape) <= 5:
                    datasetNames.append( '/' + name )    
            # Visit every group/dataset in the file            
            f.visititems(accumulateDatasetPaths)        
        return datasetNames

    def handleRowDataChange(self, changedItem ):
        """
        The user manually edited a file name in the table.
        Update the operator and other GUI elements with the new file path.
        """
        with Tracer(traceLogger):
            # Figure out which row this widget is in
            row = changedItem.row()
            column = changedItem.column()
            
            # Can't update until the row is fully initialized
            needUpdate = True
            needUpdate &= column == Column.Name or column == Column.InternalID
            needUpdate &= self.fileInfoTableWidget.item(row, Column.Name) != None 
            needUpdate &= self.fileInfoTableWidget.cellWidget(row, Column.InternalID) != None
            needUpdate &= self.fileInfoTableWidget.cellWidget(row, column) is not None
            
            if needUpdate:
                self.updateFilePath(row)

    @threadRouted    
    def updateFilePath(self, index):
        """
        Update the operator's filePath input to match the gui
        """
        with Tracer(traceLogger):
            oldLocationSetting = self.mainOperator.Dataset[index].value.location
            
            # Get the directory by inspecting the original operator path
            oldTotalPath = self.mainOperator.Dataset[index].value.filePath
            # Split into directory, filename, extension, and internal path
            lastDotIndex = oldTotalPath.rfind('.')
            extensionAndInternal = oldTotalPath[lastDotIndex:]
            extension = extensionAndInternal.split('/')[0]
            oldFilePath = oldTotalPath[:lastDotIndex] + extension
            
            fileNameText = str(self.fileInfoTableWidget.item(index, Column.Name).text())
            
            internalPathCombo = self.fileInfoTableWidget.cellWidget(index, Column.InternalID)
            #internalPath = str(self.fileInfoTableWidget.item(index, Column.InternalID).text())
            internalPath = str(internalPathCombo.currentText())
            
            directory = os.path.split(oldFilePath)[0]
            newFileNamePath = fileNameText
            if directory != '':
                newFileNamePath = directory + '/' + fileNameText
            
            newTotalPath = newFileNamePath
            if internalPath != '':
                if internalPath[0] != '/':
                    newTotalPath += '/'
                newTotalPath += internalPath
    
            cwd = self.mainOperator.WorkingDirectory.value
            absTotalPath, relTotalPath = getPathVariants( newTotalPath, cwd )
    
            # Check the location setting
            locationCombo = self.fileInfoTableWidget.cellWidget(index, Column.Location)
            comboIndex = locationCombo.currentIndex()
            newLocationSelection = locationCombo.itemData(comboIndex).toInt()[0] # In PyQt, toInt() returns a tuple
    
            if newLocationSelection == LocationOptions.Project:
                newLocationSetting = DatasetInfo.Location.ProjectInternal
            elif newLocationSelection == LocationOptions.AbsolutePath:
                newLocationSetting = DatasetInfo.Location.FileSystem
                newTotalPath = absTotalPath
            elif newLocationSelection == LocationOptions.RelativePath:
                newLocationSetting = DatasetInfo.Location.FileSystem
                newTotalPath = relTotalPath
            
            if newTotalPath != oldTotalPath or newLocationSetting != oldLocationSetting:
                # Be sure to copy so the slot notices the change when we setValue()
                datasetInfo = copy.copy(self.mainOperator.Dataset[index].value)
                datasetInfo.filePath = newTotalPath
                datasetInfo.location = newLocationSetting
        
                # TODO: First check to make sure this file exists!
                self.mainOperator.Dataset[index].setValue( datasetInfo )
        
                # Update the storage option combo to show the new path        
                self.updateStorageOptionComboBox(index, newFileNamePath)
        
    def handleRemoveButtonClicked(self):
        """
        The user clicked the "Remove" button.
        Remove the currently selected row(s) from both the GUI and the top-level operator.
        """
        with Tracer(traceLogger):
            # Figure out which dataset to remove
            rowsToDelete = set()
            selectedRanges = self.fileInfoTableWidget.selectedRanges()
            for rng in selectedRanges:
                for row in range(rng.topRow(), rng.bottomRow()+1):
                    rowsToDelete.add(row)
            
            # Remove files in reverse order so we don't have to switch indexes as we go
            for row in sorted(rowsToDelete, reverse=True):
                # Remove from the GUI
                self.fileInfoTableWidget.removeRow(row)
                # Remove from the operator input
                finalSize = len(self.mainOperator.Dataset) - 1
                self.mainOperator.Dataset.removeSlot(row, finalSize)
                
            # The gui and the operator should be in sync
            assert self.fileInfoTableWidget.rowCount() == len(self.mainOperator.Dataset)

    def handleComboSelectionChanged(self, combo, index):
        """
        Handles changes to any combo change in the table (either external path or internal path)
        """
        with Tracer(traceLogger):
            logger.debug("Combo selection changed: " + combo.itemText(1) + str(index))
    
            # Figure out which row this combo is in
            tableWidget = self.fileInfoTableWidget
            changedRow = -1
            for row in range(0, tableWidget.rowCount()):
                for column in range(Column.NumColumns):
                    widget = tableWidget.cellWidget(row, column)
                    if widget == combo:
                        changedRow = row
                        break
            assert changedRow != -1
            
            self.updateFilePath( changedRow )
                
    def handleTableSelectionChange(self):
        """
        Any time the user selects a new item, select the whole row.
        """
        with Tracer(traceLogger):
            # Figure out which dataset to remove
            selectedItemRows = set()
            selectedRanges = self.fileInfoTableWidget.selectedRanges()
            for rng in selectedRanges:
                for row in range(rng.topRow(), rng.bottomRow()+1):
                    selectedItemRows.add(row)
            
            # Disconnect from selection change notifications while we do this
            self.fileInfoTableWidget.itemSelectionChanged.disconnect(self.handleTableSelectionChange)
            for row in selectedItemRows:
                self.fileInfoTableWidget.selectRow(row)
                
            # Reconnect now that we're finished
            self.fileInfoTableWidget.itemSelectionChanged.connect(self.handleTableSelectionChange)
    














































