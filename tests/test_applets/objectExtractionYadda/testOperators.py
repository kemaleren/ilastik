import unittest
import numpy as np
from lazyflow.graph import Graph
from ilastik.applets.objectExtraction.opObjectExtraction import \
    OpObjectExtraction, OpSegmentationImage, OpObjectCounts, OpRegionFeatures

def binaryImage():
    img = np.zeros((2, 50, 50, 50, 1), dtype=np.float32)
    img[0,  0:10,  0:10,  0:10, 0] = 1
    img[0, 20:30, 20:30, 20:30, 0] = 1
    img[1, 20:30, 20:30, 20:30, 0] = 1

    return img


class TestOpSegmentationImage(unittest.TestCase):
    def setUp(self):
        g = Graph()
        self.op = OpSegmentationImage(graph=g)
        self.img = binaryImage()
        self.op.BinaryImage.setValue(self.img)

    def test_segment(self):
        segImg = self.op.SegmentationImage.value
        segImg = segImg.astype(np.int)
        self.assertEquals(segImg.shape, self.img.shape)
        self.assertTrue(segImg.min() == 0)
        self.assertTrue(segImg.max() == 2)


class TestOpObjectCounts(unittest.TestCase):
    def setUp(self):
        g = Graph()
        self.segop = OpSegmentationImage(graph=g)
        self.op = OpObjectCounts(graph=g)
        self.op.SegmentationImage.connect(self.segop.SegmentationImage)
        self.img = binaryImage()
        self.segop.BinaryImage.setValue(self.img)

    def test_count(self):
        counts = self.op.ObjectCounts[()].wait()
        self.assertEquals(len(counts), self.img.shape[0])
        self.assertEquals(counts[0], 2)
        self.assertEquals(counts[1], 1)

    def test_count_value(self):
        # for some reason, getting value returns value[0].
        counts = self.op.ObjectCounts.value
        self.assertEquals(len(counts), 2)

    # TODO: test propagateDirty and execute with roi.

class TestOpRegionFeatures(unittest.TestCase):
    def setUp(self):
        g = Graph()
        self.segop = OpSegmentationImage(graph=g)
        self.op = OpRegionFeatures(graph=g)
        self.op.SegmentationImage.connect(self.segop.SegmentationImage)
        self.img = binaryImage()
        self.segop.BinaryImage.setValue(self.img)

    def test_features(self):
        self.op.RegionFeatures.fixed = False
        # FIXME: roi specification
        feats = self.op.RegionFeatures[0, 1].wait()
        self.assertEquals(len(feats), self.img.shape[0])
        for t in feats:
            self.assertIsInstance(t, int)
            self.assertGreater(feats[t]['Count'].shape[0], 0)
            self.assertGreater(feats[t]['RegionCenter'].shape[0], 0)

        self.assertTrue(np.any(feats[0]['Count'] != feats[1]['Count']))
        self.assertTrue(np.any(feats[0]['RegionCenter'] != feats[1]['RegionCenter']))

    def test_features_value(self):
        # calling .value fails
        feats = self.op.RegionFeatures.value
        self.assertEquals(len(feats), self.img.shape[0])


    # TODO: test with roi


if __name__ == '__main__':
    unittest.main()
