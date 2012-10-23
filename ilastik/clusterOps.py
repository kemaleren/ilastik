import os
import math
import numpy
import subprocess
from lazyflow.rtype import SubRegion
from lazyflow.graph import Operator, InputSlot, OutputSlot
import itertools
import cPickle as pickle

import h5py

import logging
logger = logging.getLogger(__name__)


STATUS_FILE_NAME_FORMAT = "node status {} {}.txt"
OUTPUT_FILE_NAME_FORMAT = "node output {} {}.txt"

class OpTaskWorker(Operator):
    ScratchDirectory = InputSlot(stype='filestring')
    Input = InputSlot()
    
    ReturnCode = OutputSlot()

    def setupOutputs(self):
        self.ReturnCode.meta.dtype = bool
        self.ReturnCode.meta.shape = (1,)
    
    def execute(self, slot, subindex, roi, result):
        statusFileName = STATUS_FILE_NAME_FORMAT.format(projectFileName, str(roi) )
        outputFileName = OUTPUT_FILE_NAME_FORMAT.format(projectFileName, str(roi) )

        statusFilePath = os.path.join( self.ScratchDirectory.value, statusFileName )
        outputFilePath = os.path.join( self.ScratchDirectory.value, outputFileName )

        # Create the output file
        outputFile = h5py.File( self.OutputFile.value, 'w' )
        dataShape = numpy.subtract(roi.stop, roi.start)
        dataset = outputFile.create_dataset( 'node_result', shape=dataShape, dtype=self.Input.meta.dtype )

        # Get the result and write it to the file
        req = self.Input(roi).writeInto(dataset)
        req.wait()
        
        # Now create the status file to show that we're finished.
        statusFile = file(statusFilePath, 'w')
        statusFile.write('Yay!')
        
        result[0] = 1
        return result

    def propagateDirty(self, slot, subindex, roi):
        self.ReturnCode.setDirty( slice(None) )

class OpClusterize(Operator):
    ProjectFilePath = InputSlot(stype='filestring')
    ScratchDirectory = InputSlot(stype='filestring')
    CommandFormat = InputSlot(stype='string') # Format string for spawning a node task.
    NumJobs = InputSlot()
    Input = InputSlot()

    OutputFilePath = InputSlot()
    
    ReturnCode = OutputSlot()

    def setupOutputs(self):
        self.ReturnCode.meta.dtype = bool
        self.ReturnCode.meta.shape = (1,)
    
    def execute(self, slot, subindex, roi, result):
        # Divide up the workload into large pieces
        rois = self.getRoiList()
                
        #fullProjectFileName = self.ProjectFilePath.value.replace("/", "_").replace("\\", "_")
        # Fow now, just use the last part of the filename, without extension.
        # Later, use the fully qualified filename, in case two projects with identical names are run at the same time...
        projectFileName = os.path.split( os.path.splitext(self.ProjectFilePath.value )[0])[0]
        commandFormat = self.CommandFormat.value

        class TaskInfo():
            command = None
            statusFilePath = None
            outputFilePath = None
            subregion = None
        
        taskInfos = {}
        for roi in rois:
            roi = ( tuple(roi[0]), tuple(roi[1]) )
            taskInfo = TaskInfo()
            taskInfo.subregion = SubRegion( None, start=roi[0], stop=roi[1] )
            
            statusFileName = STATUS_FILE_NAME_FORMAT.format(projectFileName, str(roi) )
            outputFileName = OUTPUT_FILE_NAME_FORMAT.format(projectFileName, str(roi) )

            statusFilePath = os.path.join( self.ScratchDirectory.value, statusFileName )
            outputFilePath = os.path.join( self.ScratchDirectory.value, outputFileName )

            commandArgs = []
            commandArgs.append( "--project=" + self.ProjectFilePath.value )
            commandArgs.append( "--_node_work_=" + pickle.dumps( taskInfo.subregion ) )
            commandArgs.append( "--scratch_directory=" + self.ScratchDirectory.value )
            
            
            allArgs = " ".join(commandArgs)
            taskInfo.command = commandFormat.format( allArgs )
            taskInfo.statusFilePath = statusFilePath
            taskInfo.outputFilePath = outputFilePath
            taskInfos[roi] = taskInfo

            # If files are still hanging around from the last run, delete them.
            if os.path.exists( statusFilePath ):
                os.remove( statusFilePath )
            if os.path.exists( outputFilePath ):
                os.remove( outputFilePath )
            
            # Spawn the task
            logger.info("Launching node task: " + taskInfo.command )
            subprocess.call(taskInfo.command)

        # When each task completes, it creates a status file.
        while len(taskInfos) > 0:
            # TODO: Maybe replace this naive polling system with an asynchronous 
            #         file status via select.epoll or something like that.
            time.sleep(5.0)

            finished_rois = []
            destinationFile = None
            for roi, taskInfo in taskInfos.items():
                # Has the task completed yet?
                if os.path.exists( taskInfo.statusFilePath ):
                    if not os.path.exists( taskInfo.outputFilePath ):
                        raise RuntimeError( "Error: Could not locate output file from spawned task: " + taskInfo.outputFilePath )

                    # Open the file
                    f = h5py.File( taskInfo.outputFilePath, 'r' )

                    # Check the result
                    assert 'node_result' in f.keys()
                    assert f['node_result'].shape == numpy.subtract(roi[1], roi[0])
                    assert f['node_result'].dtype == self.Input.meta.dtype
                    assert f['node_result'].attrs['axistags'] == self.Input.meta.axistags.toJSON()

                    # Open the destination file if necessary
                    if destinationFile is None:
                        destinationFile = h5py.File( self.OutputFile.value )
                        if 'cluster_result' not in destinationFile.keys():
                            destinationFile.create_dataset('cluster_result', shape=self.Input.meta.shape, dtype=self.Input.meta.dtype)

                    # Copy the data into our result (which might be an h5py dataset...)
                    key = taskInfo.subregion.toSlice()
                    destinationFile['cluster_result'][key] = f['node_result'][:]
                    
                    finished_rois = roi

            # For now, we close the file after every pass in case something goes horribly wrong...
            if destinationFile is not None:
                destinationFile.close()

            # Remove the finished tasks
            for roi in finished_rois:
                del taskInfos[roi]

            result[0] = True
            return result
    
    def getRoiList(self):
        inputShape = self.Input.meta.shape
        # Use a dumb means of computing task shapes for now.
        # Find the dimension of the data in xyz, and block it up that way.
        taggedShape = self.Input.meta.getTaggedShape()

        spaceDims = filter( lambda (key, dim): key in 'xyz' and dim > 1, taggedShape.items() ) 
        numJobs = self.NumJobs.value
        numJobsPerSpaceDim = math.pow(numJobs, 1.0/len(spaceDims))
        numJobsPerSpaceDim = int(numJobsPerSpaceDim)

        print "space dims:", spaceDims
        
        roiShape = []
        for key, dim in taggedShape.items():
            if key in [key for key, value in spaceDims]:
                roiShape.append(dim / numJobsPerSpaceDim)
            else:
                roiShape.append(dim)

        roiShape = numpy.array(roiShape)
        print "RoiShape:", roiShape
        
        rois = []
        for indices in itertools.product( *[ range(0, stop, step) for stop,step in zip(inputShape, roiShape) ] ):
            start=numpy.asarray(indices)
            stop=numpy.minimum( start+roiShape, inputShape )
            rois.append( (start, stop) )

        return rois

# headless --project=myproj.ilp

    def propagateDirty(self, slot, subindex, roi):
        self.ReturnCode.setDirty( slice(None) )



































