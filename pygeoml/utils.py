import os
import numpy as np

import rasterio
from rasterio.plot import reshape_as_raster


def np_to_disk(np_array, np_fname, rpath, outdir=None):
    """
    Save a numpy array to disk using a raster filename as
    prefix

    *********

    params:
        np_array -> numpy array to save (e.g X)
        np_fname -> name for the numpy array (e.g features)
        rpath -> Full path to raster
        outdir -> output directory

    return:
        np_path -> full path of the new numpy array

    example:
        >> np_to_disk(X, 'features', myrpath, myoutdir)

    """

    basename = os.path.splitext(os.path.basename(rpath))[0]
    if not outdir:
        # set outdir at the raster location
        outdir = os.path.dirname(rpath)
    np_path = os.path.join(outdir, basename + '_' + np_fname + '.npy')
    np.save(np_path, np_array)
    return np_path


def raster_to_disk(np_array, new_rname, new_rmeta, orig_rpath, outdir=None):
    """
    Save a numpy array as geo raster to disk using the
    original raster filename as prefix

    *********

    params:
        np_array -> numpy array to save as raster
        new_rname -> name for the new raster
        new_rmeta -> metadata for the new raster
        orig_rpath -> full path of the original raster
        outdir -> output directory

    return:
        new_rpath -> full path of the new raster

    """

    prefix = os.path.splitext(os.path.basename(orig_rpath))[0]
    name = '_' + new_rname + '.gtif'
    if not outdir:
        # set outdir as the input raster location
        outdir = os.path.dirname(orig_rpath)
    new_rpath = os.path.join(outdir, prefix + name)
    with rasterio.open(new_rpath, 'w', **new_rmeta) as dst:
        dst.write(reshape_as_raster(np_array))
    return new_rpath
