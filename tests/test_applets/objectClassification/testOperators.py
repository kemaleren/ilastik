import unittest
import numpy as np
from lazyflow.graph import Graph
from ilastik.applets.objectClassification.opObjectClassification import \
    OpToImage, OpObjectTrain, OpObjectPredict
from ilastik.applets.objectExtraction.opObjectExtraction import \
    OpRegionFeatures


def segImage():
    img = np.zeros((2, 50, 50, 50, 1), dtype=np.int)
    img[0,  0:10,  0:10,  0:10, 0] = 1
    img[0, 20:25, 20:25, 20:25, 0] = 2
    img[1,  0:10,  0:10,  0:10, 0] = 1
    img[1, 10:20, 10:20, 10:20, 0] = 2
    img[1, 20:25, 20:25, 20:25, 0] = 3
    return img


class TestOpToImage(unittest.TestCase):
    def setUp(self):
        g = Graph()
        self.op = OpToImage(graph=g)

    def test(self):
        segimg = segImage()
        map_ = {0 : np.array([10, 20, 30]),
                1 : np.array([40, 50, 60, 70])}
        self.op.Image.setValue(segimg)
        self.op.ObjectMap.setValue(map_)
        img = self.op.Output.value

        self.assertEquals(img[0, 49, 49, 49, 0], 0)
        self.assertEquals(img[1, 49, 49, 49, 0], 0)
        self.assertTrue(np.all(img[0,  0:10,  0:10,  0:10, 0] == 20))
        self.assertTrue(np.all(img[0, 20:25, 20:25, 20:25, 0] == 30))
        self.assertTrue(np.all(img[1,  0:10,  0:10,  0:10, 0] == 50))
        self.assertTrue(np.all(img[1, 10:20, 10:20, 10:20, 0] == 60))
        self.assertTrue(np.all(img[1, 20:25, 20:25, 20:25, 0] == 70))


class TestOpObjectTrain(unittest.TestCase):
    def setUp(self):
        g = Graph()
        self.featsop = OpRegionFeatures(graph=g)
        self.op = OpObjectTrain(graph=g)
        self.op.Features.resize(1)
        self.op.Labels.resize(1)

        segimg = segImage()
        self.featsop.SegmentationImage.setValue(segimg)
        self.op.Features[0].connect(self.featsop.RegionFeatures)

    def test_train(self):
        labels = {0 : np.array([0, 1, 2]),
                  1 : np.array([0, 1, 1, 2])}
        self.op.Labels.setValue(labels)
        classifier = self.op.Classifier.value


class TestOpObjectPredict(unittest.TestCase):
    def setUp(self):
        g = Graph()
        self.featsop = OpRegionFeatures(graph=g)
        self.trainop = OpObjectTrain(graph=g)
        self.op = OpObjectPredict(graph=g)

        self.trainop.Features.resize(1)
        self.trainop.Labels.resize(1)

        self.op.LabelsCount.setValue(2)

        self.trainop.Features.connect(self.featsop.RegionFeatures)
        self.op.Classifier.connect(self.trainop.Classifier)

        segimg = segImage()
        labels = {0 : np.array([0, 1, 2]),
                  1 : np.array([0, 0, 0, 0,])}

        self.featsop.SegmentationImage.setValue(segimg)
        self.trainop.Features[0].connect(self.featsop.RegionFeatures)
        self.trainop.Labels.setValue(labels)

        self.op.Features.connect(self.featsop.RegionFeatures)

    def test_train(self):
        preds = self.op.Predictions[()].wait()
        self.assertTrue(np.all(preds[0] == np.array([0, 1, 2])))
        self.assertTrue(np.all(preds[1] == np.array([0, 1, 1, 2])))


    def test_train_value(self):
        v1 = self.op.Predictions.value
        v2 = self.op.Predictions[()].wait()



if __name__ == '__main__':
    unittest.main()
