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







if __name__ == "__main__":
    basedir = '/home/diego/work/dev/ess_diego/github/pygeoml/pygeoml/tests'
    fpathr1 = os.path.join(basedir,'test1.tif')
    #basedir = '/home/diego/work/dev/ess_diego/Data_Diego/Oli'
    #fpathr1 = os.path.join(basedir,'f020414t02p01r06_yuma/f020414t02p01r06_yuma','f020414t02p01r06_geo_s01.img')

    #raster = Rasterhsp(fpathr1)
    pipeline = Rproc(fpathr1)
    pipeline.load_in_memory()
    pipeline.georeference(epsg=32611, ulc_easting=741036.92, ulc_northing=3647004.6, cell_width=4.0, cell_nheight=-4.0, rotation=-13.0)
