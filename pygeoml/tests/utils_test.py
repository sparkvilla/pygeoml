import unittest
import numpy as np
from rasterio import Affine
from rasterio.crs import CRS

from pygeoml.utils import *
from pygeoml.raster import Raster

class UtilsTest(unittest.TestCase):
    # executed prior to each test
    def setUp(self):
        self.arr = np.array([[4, 4, 3, 2],
                           [4, 4, 1, 1],
                           [1, 4, 0, 2]], dtype='uint8').reshape(3, 4, 1)

        self.masked_filled = np.array([[4, 4, 0, 0],
                           [4, 4, 0, 0],
                           [0, 4, 0, 0]], dtype='uint8').reshape(3, 4, 1)

        self.arr4ch = np.array([[[4, 4, 3, 2],
                           [4, 4, 1, 1],
                           [1, 4, 0, 2]],
                           [[4, 4, 3, 2],
                           [4, 4, 1, 1],
                           [1, 4, 0, 2]]], dtype='uint8')

        self.masked_filled4ch = np.array([[[4, 4, 0, 0],
                           [4, 4, 0, 0],
                           [0, 4, 0, 0]],
                           [[4, 4, 0, 0],
                           [4, 4, 0, 0],
                           [0, 4, 0, 0]]], dtype='uint8')

        self.meta = {'driver': 'GTiff',
                      'dtype': 'uint8',
                      'nodata': None,
                      'width': 4,
                      'height': 3,
                      'count': 1,
                      'crs': CRS.from_epsg(32737),
                      'transform': Affine(10.0, 0.0, 199980.0,
                                                  0.0, -10.0, 9300040.0)}


    # executed after each test
    def tearDown(self):
        pass


    def test_mask_and_fill(self):
        # create a mask array
        ma = Raster.mask_arr_equal(self.arr, [1, 2, 3])
        # use a mask to mask an arr
        maf = mask_and_fill(self.arr, ma.mask)
        self.assertEqual(np.array_equal(maf, self.masked_filled), True)

    def test_mask_and_fill_4ch(self):
        ma = Raster.mask_arr_equal(self.arr4ch, [1, 2, 3])
        maf = mask_and_fill(self.arr4ch, ma.mask)
        self.assertEqual(np.array_equal(maf, self.masked_filled4ch), True)

    def test_raster_to_disk(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        outdir = os.path.join(dir_path, 'out')
        rpath = raster_to_disk(self.arr, 'test_raster', self.meta, outdir)
        r = Raster(rpath)
        arr = r.load_as_arr()
        self.assertEqual(np.array_equal(arr, self.arr), True)

    def test_raster_to_disk_masked(self):
        ma = Raster.mask_arr_equal(self.arr, [1, 2, 3])
        maf = mask_and_fill(self.arr, ma.mask)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        outdir = os.path.join(dir_path, 'out')
        rpath = raster_to_disk(maf, 'test_raster_masked_filled', self.meta, outdir)
        r = Raster(rpath)
        arr = r.load_as_arr()
        self.assertEqual(np.array_equal(arr, self.masked_filled), True)


if __name__ == '__main__':
    unittest.main()
