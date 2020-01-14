import os
import copy
import glob

import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster, plotting_extent
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.crs import CRS

import numpy as np

from raster import Rasterhsp

# Not working properly

class Rproc:

    def __init__(self, path_to_raster=None, outdir=None):

        self.path_to_raster = path_to_raster
        self.outdir = outdir
        self.memfile = MemoryFile()

        with rasterio.open(path_to_raster) as dataset:
            self.driver = dataset.driver
            self.count = dataset.count
            self.desc = dataset.descriptions
            self.crs = dataset.crs
            self.meta = dataset.meta
            self.height = dataset.height
            self.width = dataset.width
            self.transform = dataset.transform
            self.bounds = dataset.bounds

    def load_in_memory(self):
        with rasterio.open(self.path_to_raster) as src:
            # make a copy of the raster data and metadata in memory
            data = self.memfile.open(**src.profile)
            data.write(src.read(out_shape=(src.count, src.height, src.width)))
            data.close()

    def save_raster(self):
        pass

    def georeference(self, epsg, ulc_easting, ulc_northing, cell_width, cell_nheight, rotation=0):
        """
        Uses a scene classification file (20m or 60m resolution) to build a mask array

        ************

        args:
            filepath -- Full path to scf file
        """

        # Build transform and crs attributes
        transform = Affine(cell_width, rotation, ulc_easting, rotation, cell_nheight, ulc_northing)
        crs = CRS.from_epsg(epsg)

        # change metadata
        metadata = self.meta
        metadata.update(transform=transform, driver='GTiff', crs=crs)

        # update metadata in memory
        update_data = self.memfile.open('w', **metadata)
        update_data.write()
        update_data.close()



def get_ds(path):
    with rasterio.open(path) as data_orig:
        print(repr(data_orig))
        memfile = MemoryFile()
        metadata = data_orig.meta
        data_proc = memfile.open(**metadata)
        print(repr(data_proc))
        data_proc.write(data_orig)
    return data_proc



if __name__ == "__main__":
    basedir = '/home/diego/work/dev/ess_diego/github/pygeoml/pygeoml/tests'
    fpathr1 = os.path.join(basedir,'test1.tif')
    #basedir = '/home/diego/work/dev/ess_diego/Data_Diego/Oli'
    #fpathr1 = os.path.join(basedir,'f020414t02p01r06_yuma/f020414t02p01r06_yuma','f020414t02p01r06_geo_s01.img')

    #raster = Rasterhsp(fpathr1)
    pipeline = Rproc(fpathr1)
    pipeline.load_in_memory()
    pipeline.georeference(epsg=32611, ulc_easting=741036.92, ulc_northing=3647004.6, cell_width=4.0, cell_nheight=-4.0, rotation=-13.0)
