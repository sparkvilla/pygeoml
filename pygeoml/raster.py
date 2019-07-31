import os
import copy

import rasterio
import rasterio.plot
from rasterio.mask import mask
from rasterio.merge import merge
import numpy as np

from shapely.geometry import mapping

class Rasterobj():

    RED = 672
    NIR = 814

    def __init__(self, path_to_raster, outdir=None):

        self.path_to_raster = path_to_raster
        self.outdir = outdir

        with rasterio.open(path_to_raster) as dataset:
            self.driver = dataset.driver
            self.num_bands = dataset.count
            self.desc = dataset.descriptions
            self.crs = dataset.crs
            self.meta = dataset.meta
            self.height = self.meta['height']
            self.width = self.meta['width']
            self.band_wl = {}
            self.red_band = None
            self.nir_band = None
            self.ndvi_band = None
            self.wl = None

        ## To be replaced with parser for specific driver ##
        if self.driver == 'ENVI':
            print("Found {} driver. Start band -> wl mapping...".format(self.driver))
            table = str.maketrans(dict.fromkeys("()"))
            # remap actual band number with wavelength
            for idx in range(self.num_bands):
                l = self.desc[idx].translate(table).split()
                key = idx+1 # start from 1
                # To access the band number written in desc:
                # key = l[6].split(":")[0]
                val = round(float(l[-2]), 3)
                self.band_wl[key] = val
            self.wl = self.band_wl.values()

        if self.driver == 'GTiff':
            print("Found {} driver. Start band -> wl mapping...".format(self.driver))

    @staticmethod
    def _get_idx_band_at_wl(mapping, wl):
        """
        Get the nearest wl value among the values of mapping
        and return the corresponding key

        example:
        >> print(Dataset._get_bands_at_wl({1:587.67,2:597.17,3:606.85}, 600))
        >> 2
        """
        differences = {key:abs(wl-val) for key, val in mapping.items()}
        idx_band = min(differences, key=differences.get)
        return idx_band

    @staticmethod
    def _average_bands(arr, indices):
        """
        This function takes a numpy array
        rank=3 (bands,height,width) and indices along 0 axis

        It builds a list of arrays (bands) using indices

        It returns a numpy array element-wise average of the arrays
        numpy array rank=2 (height,width)
        """
        arrs = [arr[i,:,:] for i in indices]
        return np.mean(arrs,axis=0)

    def get_pixel_array(self, col_start):
        """
        Get a one pixel window

        Return a 1D np array of len equal to self.wl

        """
        pass

    def get_red_and_nir_bands(self):
        """
        It fills the red and nir instance attributes with
        the respective band, i.e. 2D np-array extracted from the raster
        """
        red_idx = Rasterobj._get_idx_band_at_wl(self.band_wl, self.RED)
        nir_idx = Rasterobj._get_idx_band_at_wl(self.band_wl, self.NIR)
        with rasterio.open(self.path_to_raster) as dataset:
            # +1 -> indexing: numpy start at 0 rasterio at 1
            self.red_band = dataset.read(red_idx)
            self.nir_band = dataset.read(nir_idx)

    def get_ndvi_band(self, write=False):
        """
        It fills the ndvi instance attribute with a 2D np-array
        calculated the using the red and nir bands attributes.

        When write=True the ndvi is saved to disk as ndvi.gtif

        """
        if (isinstance(self.red_band,np.ndarray) & isinstance(self.nir_band,np.ndarray)):
            np.seterr(divide='ignore', invalid='ignore')
            self.ndvi_band = (self.nir_band.astype(float)-self.red_band.astype(float))/(self.nir_band.astype(float)+self.red_band.astype(float))

            # write to disk
            if write:
                ndvi_meta = copy.deepcopy(self.meta)
                ndvi_meta.update(count=1, dtype="float64", driver='GTiff')
                ndvi_path = os.path.dirname(self.path_to_raster)
                if self.outdir:
                    ndvi_path = self.outdir
                with rasterio.open(os.path.join(ndvi_path,'ndvi.gtif'), 'w', **ndvi_meta) as dst:
                            dst.write(self.ndvi_band, 1)
        else:
            raise ValueError("Red and Nir band values are not correct. Try to call first 'get_red_and_nir_bands()' method.")

