import os
import numpy as np
import csv
import json
import math

import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image

import pdb

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


def raster_to_disk(np_array, new_rname, new_rmeta, outdir):
    """
    Save a numpy array as geo-raster to disk

    'p_rpath' param uses a second raster filename as prefix

    *********

    params:
        np_array ->  3D numpy array to save as raster
        new_rname -> name for the new raster
        new_rmeta -> metadata for the new raster
        outdir -> output directory

    return:
        new_rpath -> full path of the new raster

    """
    assert (new_rmeta['driver'] == 'GTiff'),\
        "Please use GTiff driver to write to disk. \
    Passed {} instead.".format(new_rmeta['driver'])

    assert (np_array.ndim == 3),\
        "np_array must have ndim = 3. \
Passed np_array of dimension {} instead.".format(np_array.ndim)

    name = new_rname + '.tif'
    new_rpath = os.path.join(outdir, name)

    with rasterio.open(new_rpath, 'w', **new_rmeta) as dst:
        dst.write(reshape_as_raster(np_array))
    return new_rpath

def aoi_xy_to_disk(rpath_orig, rname, window, outdir):
    """
    Writes an area of interest (aoi) to disk
    *********

    params:
        rpath ->  Full path to original raster
        window -> rasterio.windows.Window
        rnam -> name aoi the aoi to write to disk
        outdir -> output directory

    return:
        new_rpath -> full path of the new raster
    """
    name = rname + '.tif'
    new_rpath = os.path.join(outdir, name)

    with rasterio.open(rpath_orig) as src:
        new_meta = src.meta.copy()
        new_meta.update({
            'driver': 'GTiff',
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})

        with rasterio.open(new_rpath, 'w', **new_meta) as dst:
            dst.write(src.read(window=window))
    return os.path.abspath(new_rpath)


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

    name = new_rname + '.tif'
    new_rpath = os.path.join(outdir, name)
    with rasterio.open(new_rpath, 'w', **new_rmeta) as dst:
        for _id, fl in enumerate(rfiles, start=1):
            with rasterio.open(fl) as src1:
                np_arr = reshape_as_image(src1.read())
                if mask is not None:
                    np_arr = mask_and_fill(np_arr, mask)
                    dst.write_band(_id, np_arr[:, :, 0].astype(new_rmeta['dtype']))
    return new_rpath


def mosaic_to_disk(rfiles, new_rname, outdir):
    """
    Merge more rasters, band by band, to form a mosaic and
    save it to disk
    *********

    params:
        rfiles -> list of rasters path to be merged
        new_rname -> name of the final mosaic
        outdir -> output directory

    """

    src_files = [rasterio.open(raster) for raster in rfiles]

    first_src = src_files[0]
    first_res = first_src.res
    dtype = first_src.dtypes[0]
    # Determine output band count
    output_count = first_src.count


    # Extent of all inputs
    # scan input files
    xs = []
    ys = []
    for src in src_files:
        pdb.set_trace()
        left, bottom, right, top = src.bounds
        xs.extend([left, right])
        ys.extend([bottom, top])
    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    out_transform = rasterio.Affine.translation(dst_w, dst_n)

    # Resolution/pixel size
    res = first_res
    out_transform *= rasterio.Affine.scale(res[0], -res[1])

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    output_width = int(math.ceil((dst_e - dst_w) / res[0]))
    output_height = int(math.ceil((dst_n - dst_s) / res[1]))

    # Adjust bounds to fit
    dst_e, dst_s = out_transform * (output_width, output_height)
    # create destination array
    # destination array shape
    shape = (output_height, output_width)
    # dest = np.zeros((output_count, output_height, output_width), dtype=dtype)
    # Using numpy.memmap to create arrays directly mapped into a file
    from tempfile import mkdtemp
    memmap_file = os.path.join(mkdtemp(), 'test.mymemmap')
    dest_array = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=shape)

    dest_profile = {
            "driver": 'GTiff',
            "height": dest_array.shape[0],
            "width": dest_array.shape[1],
            "count": output_count,
            "dtype": dest_array.dtype,
            "crs": '+proj=latlong',
            "transform": out_transform
    }

    name = new_rname + '.tif'
    new_rpath = os.path.join(outdir, name)
    # open output file in write/read mode and fill with destination mosaic array
    with rasterio.open(new_rpath, 'w+', **dest_profile) as mosaic_raster:
        for src in src_files:
            for ji, src_window in src.block_windows():
                print(ji)
                arr = src.read(window=src_window)
                # store raster nodata value
                nodata = src.nodatavals[0]
                # replace zeros with nan
                #arr[arr == nodata] = np.nan
                # convert relative input window location to relative output # windowlocation
                # using real world coordinates (bounds)
                src_bounds = rasterio.windows.bounds(src_window, transform=src.profile["transform"])
                dst_window = rasterio.windows.from_bounds(*src_bounds, transform=mosaic_raster.profile["transform"])

                # round the values of dest_window as they can be float
                dst_window = rasterio.windows.Window(round(dst_window.col_off), round(dst_window.row_off), round(dst_window.width), round(dst_window.height))
                # before writing the window, replace source nodata with dest
                # nodataas it can already have been written (e.g. another adjacent # country)
                # https://stackoverflow.com/a/43590909/1979665
                dest_pre = mosaic_raster.read(window=dst_window)
                mask = (arr == nodata)
                r_mod = np.copy(arr)
                r_mod[mask] = dest_pre[mask]
                mosaic_raster.write(r_mod, window=dst_window)

    os.remove(memmap_file)
    return new_rpath

# Serializing Python Objects not supported by JSON
class NumpyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.str_):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)


def load_json(fpath):
    """
    Load a json object from disk
    *********

    params:
        json_arr -> full path to json file

    return:
        a list of dictionaries
    """

    with open(fpath, 'r', encoding='utf-8') as f:
        json_str = json.loads(f.read())

    data = json.loads(json_str)
    return data


def to_json(datadict, encoder=json.JSONEncoder):
    """
    Serialize python objects using json. Supports
    numpy.ndarray serialization when encoder=NumpyEncoder
    *********

    params:
        datadict -> a dict with the python object to serialize
                   e.g. {'obj_name': obj, ..}
        encoder -> json Encoder. To be replaced by a different
                   encoder, e.g. NumpyEncoder for numpy.ndarray, to
                   serialize datatypes not supported by the default

    return:
        a json object

    """
    return json.dumps(datadict, cls=encoder)


def json_to_disk(json_arr, name, outdir):
    """
    Write a json object to disk
    *********

    params:
        json_arr -> json object
        name -> filename to write
    """
    fname = name + '.json'
    fpath = os.path.join(outdir, fname)
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(json_arr, f)
    return fpath


def to_csv(fpath, data):
    with open(fpath, 'a') as fdata:
        writer = csv.writer(fdata, delimiter=',')
        for row in data:
            writer.writerows(row)


if __name__ == "__main__":


    basedir = '/home/diego/work/dev/imgs/for_mosaic'

    #search_criteria = "*.tif"
    # output base directory
    #fl = os.path.join(basedir, search_criteria)

    #import glob
    #rf = glob.glob(fl)
    #mosaic_to_disk(rf, 'mosaic', basedir)
