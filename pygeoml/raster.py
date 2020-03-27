import os
import copy
import sys
import rasterio
import rasterio.plot
from rasterio.plot import reshape_as_image, reshape_as_raster, plotting_extent
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.crs import CRS

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from pygeoml.utils import raster_to_disk, mask_and_fill, stack_to_disk
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Raster:
    """
    Base raster class to process geo-referenced raster data.
    """

    def __init__(self, path_to_raster):

        self.path_to_raster = path_to_raster

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
            self.nodata = dataset.nodata
            self.nodatavals = dataset.nodatavals
            self.block_shapes = dataset.block_shapes
            self.files = dataset.files
            self.indexes = dataset.indexes
            self.name = dataset.name
            self.res = dataset.res

    def __str__(self):
        return "Raster(height: {}, width: {}, bands: {})".format(self.height, self.width, self.count)

    def __repr__(self):
        return 'Raster({})'.format(self.path_to_raster)

    def load_as_arr(self, **kwargs):
        """
        (to be modified.)
        ( 1. change to take a rasterio.windows.Window instance as args)

        Load a raster image as a 3D numpy array. The axis of the array have the order:

        (height, width, bands) or
        (rows, cols, channels)

        This is the order expected by image processing and visualization software;
        i.e. matplotlib, scikit-image, etc..

        Use 'heigh' and 'width' to load only a window of the raster
        and 'band' to select a single or multiple bands of choice.


        keyword args:
            height -- raster height (default full height)
            width -- raster width (default full width)
            bands -- number of bands to load, for multiple bands
                     use [1,2,3] (default all bands)
            col_off -- starting column (default 0)
            row_off -- starting row (default 0)

        """
        # all bands by defauls
        bands = [i for i in range(1, self.count+1)]

        height, width, bands, col_off, row_off, masked = kwargs.get('height',self.height), kwargs.get('width',self.width),\
                                                           kwargs.get('bands', bands), kwargs.get('col_off', 0),kwargs.get('row_off', 0),\
                                                           kwargs.get('masked',False)
        # Even by passing an int for index
        # always return a 3D np array
        if not isinstance(bands, list):
                bands = [bands]

        with rasterio.open(self.path_to_raster) as dataset:
            img = dataset.read(bands, window=Window(col_off, row_off, width, height), masked=masked)
        return reshape_as_image(img)

    def transform_to_coordinates(self, rows, cols):
        """
        It returns geo coordinates at row,col indexes
        """
        xs,ys = rasterio.transform.xy(self.transform, rows, cols)
        return xs, ys

    def transform_to_row_col(self, xs, ys):
        """
        It returns row,col at geo coordinates indexes
        """
        row,col = rasterio.transform.rowcol(self.transform, xs, ys)
        return row, col

    def get_patches(self, v_split, h_split):
        """
        Yields patches of the entire raster as numpy array.

        params:
            v_split -> number of times the raster will be divided
                       along its height
            h_split -> number of times the raster will be divided
                       along its width
        """
        # get patch arrays
        v_arrays = np.array_split(np.arange(self.height), v_split)
        h_arrays = np.array_split(np.arange(self.width), h_split)

        for v_arr in v_arrays:
            v_start = v_arr[0]
            for h_arr in h_arrays:
                h_start = h_arr[0]
                arr = self.load_as_arr(height=len(v_arr), width=len(h_arr),
                                       col_off=h_start, row_off=v_start)
                yield (v_start, h_start, arr)

    def load_spectral_profile(self, row, col):
        """
        Get a one pixel window along the stack

        *********
        params:
            row -> row index
            col -> col index

        ********
        return:
            1D numpy array of len equal to self.wl

        """
        return self.load_as_arr(height=1, width=1, col_off=col, row_off=row)[0,0,:]

    @classmethod
    def mask_arr_equal(cls, arr, vals):
        """
        Mask the values of an array that are  equal to a given value.

        *********

        params:
            arr -> 3D numpy array (rows, cols, single channel)
            vals -> a list of values

        """
        return np.ma.MaskedArray(arr, np.in1d(arr, vals))


    @classmethod
    def calc_ndvi(cls, red_obj, nir_obj, outdir):
        """
        *********

        params:
            red_arr -> pygeoml.Raster
            nir_arr -> pygeoml.Raster
            outdir -> full path to output directory

        """
        logger.debug('Calculate ndvi with red: {} and nir: {}'.format(red_obj, nir_obj))
        red_arr = red_obj.load_as_arr()[:,:,0]
        nir_arr = nir_obj.load_as_arr()[:,:,0]

        np.seterr(divide='ignore', invalid='ignore')
        ndvi_arr = (nir_arr.astype(np.float32)-red_arr.astype(np.float32))/(nir_arr.astype(np.float32)+red_arr.astype(np.float32))

        # add a third axes (channel) to the np array
        ndvi_arr = np.expand_dims(ndvi_arr, axis=2)
        # grab and copy metadata of one of the two array
        ndvi_meta = copy.deepcopy(red_obj.meta)
        ndvi_meta.update(count=1, dtype="float32", driver='GTiff')

        rpath = raster_to_disk(ndvi_arr, 'ndvi', ndvi_meta, outdir)

        return Raster(rpath)


    @classmethod
    def resample_raster(cls, r_obj, scale=2, outdir=None):
        """
        Change the cell size of an existing raster object.

        Can be used for both:

        Upsampling; converting to higher resolution/smaller cells
        Downsampling converting to lower resolution/larger cells

        a raster object.

        Save the new raster directly to disk.

        ************

        params:
            r_obj -> An istance of the Raster class
            scale -> scaling factor to change the cell size with.
                     scale = 2 -> Upsampling e.g from 10m to 20m resolution
                     scale = 0.5 -> Downsampling e.g from 20m to 10m resolution
            outdir -> full path to output directory

        return:
            a Raster instance of the resampled raster.
        """
        t = r_obj.transform

        # rescale the metadata
        transform = Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)
        height = r_obj.height * scale
        width = r_obj.width * scale

        new_meta = copy.deepcopy(r_obj.meta)
        new_meta.update(transform=transform, driver='GTiff', height=height, width=width)

        # UP-sampling
        with rasterio.open(r_obj.path_to_raster) as dataset:
            r_arr = dataset.read(out_shape=(dataset.count, height , width), resampling=Resampling.nearest)
            np_arr = reshape_as_image(r_arr)

        rpath = raster_to_disk(np_arr, 'resampled', new_meta,
                        outdir)

        return Raster(rpath)

    @classmethod
    def create_stack(cls, rfiles, outdir, dtype=None, mask=None):
        """
        Create a stack of rasters

        ************

        params:
            rfiles -> list of raster paths to be stacked
            outdir -> full path to output directory
            mask -> 3D boolean masked array

        return:
            a Raster instance of the stack
        """

        # Read metadata of first file and assume all other bands are the same
        with rasterio.open(rfiles[0]) as src0:
            meta = src0.meta
            # Update  metadata to reflect the number of layers
            meta.update(count = len(rfiles))
            meta.update(driver = 'GTiff')
            if dtype:
                meta.update(dtype = dtype)
        stack_fname = 'multibands'
        if mask is not None:
            stack_fname = 'multibands_masked'
        stack_path = stack_to_disk(rfiles, stack_fname, meta, outdir, mask)
        return Raster(stack_path)

    @classmethod
    def stitch(cls, r_obj_up, r_obj_down, axis=0, write=False, outdir=None):
        """
        (to be tested)
        """

        height = r_obj_up.height + r_obj_down.height
        width = r_obj_up.width

        new_meta = copy.deepcopy(r_obj_up.meta)
        new_meta.update(driver='GTiff', height=height, width=width)

        arr_up = r_obj_up.load_as_arr()
        arr_down = r_obj_down.load_as_arr()

        arr_final = np.concatenate((arr_up, arr_down), axis=axis)

        if write:
            fname = os.path.basename(r_obj_up.path_to_raster).split('.')[0]+'_stitched'
            if not outdir:
                # set outdir as the input raster location
                outdir = os.path.dirname(r_obj_up.path_to_raster)

            with rasterio.open(os.path.join(outdir,fname+'.gtif'), 'w', **new_meta) as dst:
                dst.write(reshape_as_raster(arr_final))
        return arr_final

    @classmethod
    def georeference_raster(cls, r_obj, epsg, ulc_easting, ulc_northing, cell_width, cell_nheight, rotation=0, outdir=None):
        """
        (to be tested)
        """

        # Build transform and crs attributes
        transform = Affine(cell_width, rotation, ulc_easting, rotation, cell_nheight, ulc_northing)
        crs = CRS.from_epsg(epsg)

        new_meta = copy.deepcopy(r_obj.meta)
        new_meta.update(transform=transform, driver='GTiff', crs=crs)

        with rasterio.open(r_obj.path_to_raster) as dataset:
            r_arr = dataset.read()

        fname = os.path.basename(r_obj.path_to_raster).split('.')[0]+'_georeferenced'
        if not outdir:
            # set outdir as the input raster location
            outdir = os.path.dirname(r_obj.path_to_raster)

        with rasterio.open(os.path.join(outdir,fname+'.gtif'), 'w', **new_meta) as dst:
            dst.write(r_arr)
        return Raster(os.path.join(outdir,fname+'.gtif'))

    @classmethod
    def merge_rasters(cls, paths):
        """
        (to be tested)
        """
        if paths:
            src_files_to_mosaic = []
            paths_new = paths[:]

        for path in paths_new:
            src = rasterio.open(path)
            src_files_to_mosaic.append(src)

        mosaic, out_trans = merge(src_files_to_mosaic, indexes=[40,41])
        return mosaic, out_trans

    @staticmethod
    def _normalize(arr):
        """Normalizes numpy arrays into scale 0.0 - 1.0"""
        array_min, array_max = arr.min(), arr.max()
        return ((arr - array_min)/(array_max - array_min))


class Rasterhsp(Raster):
    """
    (to be modified)
    """

    RED = 672
    NIR = 814

    def __init__(self, path_to_raster):
        super().__init__(path_to_raster)
        assert self.count > 1, '"Rasterhsp" class process multibands raster only, your raster has a single band. Try to use the "Raster" class instead.'
        self.wl = None
        self.band_wl = {}

    def map_band_wl(self):
        ## This is case specific. It has to be replaced with a more general version. ##
        ## To be replaced with parser for specific driver ##
        if self.driver == 'ENVI':
            print("Found {} driver. Start band -> wl mapping...".format(self.driver))
            table = str.maketrans(dict.fromkeys("()"))
            # remap actual band number with wavelength
            for idx in range(self.count):
                l = self.desc[idx].translate(table).split()
                key = idx+1 # start from 1
                # To access the band number written in desc:
                # key = l[6].split(":")[0]
                val = round(float(l[-2]), 3)
                self.band_wl[key] = val
            self.wl = list(self.band_wl.values())

        if self.driver == 'GTiff':
            print("Found {} driver. Start band -> wl mapping...".format(self.driver))

    def __str__(self):
        return "Raster(height: {}, width: {}, bands: {})".format(self.height, self.width, self.num_bands)

    @staticmethod
    def _get_idx_band_at_wl(mapping, wl):
        """
        Get the nearest wl value among the values of mapping
        and return the corresponding key

        example:
        >> print(Rasterhsp._get_idx_band_at_wl({1:587.67,2:597.17,3:606.85}, 600))
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


