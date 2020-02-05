import os
import numpy as np

import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image


def mask_and_fill(arr, mask, fill_value=0):
    """
    Mask an array using a mask array and fill it with
    a fill value.

    *********

    params:
        arr -> 3D numpy array to be masked (rows, cols, channels)
        mask -> 3D boolean masked array

    return:
        masked_arr_filled -> masked 3D numpy array
    """
    # check arr and mask have the same dim
    assert (arr.shape == mask.shape),\
        "Array and mask must have the same dimensions!"
    masked_arr = np.ma.array(arr, mask=mask)

    # Fill masked vales with zero !! maybe to be changed
    masked_arr_filled = np.ma.filled(masked_arr, fill_value=fill_value)

    return masked_arr_filled


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


def raster_to_disk(np_array, new_rname, new_rmeta, outdir, p_rpath=None):
    """
    Save a numpy array as geo-raster to disk

    'p_rpath' param uses a second raster filename as prefix

    *********

    params:
        np_array ->  3D numpy array to save as raster
        new_rname -> name for the new raster
        new_rmeta -> metadata for the new raster
        outdir -> output directory
        p_rpath -> full path of the original raster

    return:
        new_rpath -> full path of the new raster

    """
    assert (new_rmeta['driver'] == 'GTiff'),\
        "Please use GTiff driver to write to disk. \
    Passed {} instead.".format(new_rmeta['driver'])

    name = new_rname + '.gtif'
    new_rpath = os.path.join(outdir, name)

    if p_rpath:
        prefix = os.path.splitext(os.path.basename(p_rpath))[0]
        new_rpath = os.path.join(outdir, prefix + '_' + name)

    with rasterio.open(new_rpath, 'w', **new_rmeta) as dst:
        dst.write(reshape_as_raster(np_array))
    return new_rpath


def stack_to_disk(rfiles, new_rname, new_rmeta, outdir, mask=None):
    """
    Stack several rasters layer on a single raster and
    save it to disk
    *********

    params:
        rfiles -> list of raster path to be stacked
        new_rname -> name of the final stack
        new_rmeta -> metadata of the final raster
        outdir -> output directory
        mask -> 3D boolean masked array

    """
    assert (new_rmeta['driver'] == 'GTiff'),\
        "Please use GTiff driver to write to disk. \
    Passed {} instead.".format(new_rmeta['driver'])

    name = new_rname + '.gtif'
    new_rpath = os.path.join(outdir, name)
    with rasterio.open(new_rpath, 'w', **new_rmeta) as dst:
        for _id, fl in enumerate(rfiles, start=1):
            with rasterio.open(fl) as src1:
                np_arr = reshape_as_image(src1.read())
                if mask is not None:
                    np_arr = mask_and_fill(np_arr, mask)
                    dst.write_band(_id, np_arr[:, :, 0])
    return new_rpath
