import os
import sys

import rasterio
from rasterio import Affine, MemoryFile
from rasterio.enums import Resampling
from rasterio.crs import CRS

import numpy as np

import logging
import pdb


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_blocks(src, **kwargs):
    """
    Params:
    *******
    src -> rasterio DataReader

    keyword args:
    *******
    bidx -- The band index (using 1-based indexing) from
            which to extract windows. default (src.count); i.e.
            all bands

    Yield:
    *******
    A window tuple, e.g Window(col_off=0, row_off=0, width=791, height=3)
    that can be used in calls to Read or Write

    The unused '_' refers to (row, col) indexes of this block
    within all blocks of the dataset.
    """
    bidx = kwargs.get('bidx', src.count),
    block_shape = src.block_shapes[0]
    #pdb.set_trace()
    logger.debug('Shape of blocks for any band is (height, width): {}'.format(block_shape))
    for _, window in src.block_windows(bidx = bidx[0]):
        yield window


def crop_raster(src, window):
    """
    Crop a defined window from a geo-raster

    Params:
    *******
    window -> rasterio.windows.Window

    Return:
    *******
    src -> rasterio.io.DataReader

    """
    # load window
    r_arr = src.read(window=window) # (channels,rows,cols)

    # update metadata
    src.height = r_arr.shape[1]
    src.width = r_arr.shape[2]
    # 3 load window and save to dist as raster
    # 4 use memoryfile to update Datareader and pass it along the next step
    pass

if __name__ == "__main__":
    basedir = '/home/diego/work/dev/ess_diego/python/Rasterio_features_test'
    fpath = os.path.join(basedir,'T37MBN_20190628T073621_TCI_10m.jp2')

    with rasterio.open(fpath) as src:
        for blk in get_blocks(src):
            print(blk)
