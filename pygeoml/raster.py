import os
import copy
import glob
import re
#from parser import Sentinel2
import rasterio
import rasterio.plot
from rasterio.plot import reshape_as_image
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.windows import Window
import numpy as np

from shapely.geometry import mapping


class Raster:
    """
    Base raster class to process geo data.
    """

    def __init__(self, path_to_raster, outdir=None):

        self.path_to_raster = path_to_raster
        self.outdir = outdir

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

    def __str__(self):
        return "Raster(height: {}, width: {}, bands: {})".format(self.height, self.width, self.count)

    def __repr__(self):
        return 'Rastermps({})'.format(self.path_to_raster)

    def load_as_arr(self, height=None, width=None, bands=None, col_off=0, row_off=0):
        """
        Return a raster as a 3D numpy array. The axis of the array have the order:

        (height, width, bands) or
        (rows, cols, channels)

        This is the order expected by image processing and visualization software;
        i.e. matplotlib, scikit-image, etc..

        By default it load the entire raster, else one can specify a window.
        """
        # check whether to load the entire array
        if height is None:
            height = self.height
        if width is None:
            width = self.width
        if bands is None:
            bands = [i for i in range(1, self.count+1)]

        with rasterio.open(self.path_to_raster) as dataset:
            img = dataset.read(bands, window=Window(col_off, row_off, width, height))
        return reshape_as_image(img)

    @classmethod
    def load_rgb(cls, red_obj, green_obj, blue_obj, height, width, bands=[1], col_off=0,row_off=0):

        r = red_obj.load_as_arr(height, width, bands, col_off, row_off)
        g = green_obj.load_as_arr(height, width, bands, col_off, row_off)
        b = blue_obj.load_as_arr(height, width, bands, col_off, row_off)


        rgb_norm = np.empty((height,width,3), dtype=np.float32)
        rgb_norm[:,:,0] = Raster.normalize(r[:,:,0])
        rgb_norm[:,:,1] = Raster.normalize(g[:,:,0])
        rgb_norm[:,:,2] = Raster.normalize(b[:,:,0])

        # normalized RGB natural color composite
        return rgb_norm


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



    def write_tiles(self, tile_size_x=50, tile_size_y=70): # To be finished
        with rasterio.open(self.path_to_raster) as dataset:
            for col in range(0, self.width, tile_size_x):
                for row in range(0, self.height, tile_size_y):
                    #print(row,col)
                    tile = dataset.read(window=Window(col, row, tile_size_x, tile_size_y))
                    yield tile.shape

    @classmethod
    def load_ndvi(cls, red_obj, nir_obj, height, width, bands=[1], col_off=0, row_off=0, write=False):
        """
        Return a calculated ndvi 3D numpy array. The axis of the array have the order:

        (height, width, bands) or
        (rows, cols, channels)

        This is the order expected by image processing and visualization software;
        i.e. matplotlib, scikit-image, etc..

        By default it load the entire raster, else one can specify a window.

        When write=True the ndvi is saved to disk as ndvi.gtif

        """
        red_arr = red_obj.load_as_arr(height, width, bands, col_off, row_off)[:,:,0]
        nir_arr = nir_obj.load_as_arr(height, width, bands, col_off, row_off)[:,:,0]

        np.seterr(divide='ignore', invalid='ignore')
        ndvi_arr = (nir_arr.astype(float)-red_arr.astype(float))/(nir_arr.astype(float)+red_arr.astype(float))

        # write to disk
        if write:
            # grab and copy metadata of one of the two array
            ndvi_meta = copy.deepcopy(red_obj.meta)
            ndvi_meta.update(count=1, dtype="float64", driver='GTiff')
            ndvi_path = os.path.dirname(red_obj.path_to_raster)
            with rasterio.open(os.path.join(ndvi_path,'ndvi.gtif'), 'w', **ndvi_meta) as dst:
                dst.descriptions = ['ndvi']
                dst.write(ndvi_arr, 1)
        return ndvi_arr

    @staticmethod
    def normalize(arr):
        """Normalizes numpy arrays into scale 0.0 - 1.0"""
        array_min, array_max = arr.min(), arr.max()
        return ((arr - array_min)/(array_max - array_min))

class Rastermsp(Raster):

    def __init__(self, path_to_raster):
        super().__init__(path_to_raster)
        assert self.count > 1, '"Rastermsp" class process multibands raster only, your raster has a single band. Try to use the "Raster" class instead.'

    @classmethod
    def create_stack(cls, dirpath, parser, name='multibands'):

        parse = parser(dirpath)

        fname = name+'_'+parse.date+'.gtif'

        #filepath for image we're writing out
        img_fp = os.path.join(dirpath, fname)

        # get raster files and tags
        srfiles = [ i[0] for i in parse.srfiles]
        srfiles_tags = [ i[1] for i in parse.srfiles]
        # Read metadata of first file and assume all other bands are the same
        with rasterio.open(srfiles[0]) as src0:
            meta = src0.meta
            # Update  metadata to reflect the number of layers
            meta.update(count = len(srfiles))
            # Read # each # layer # and # write # it # to # stack
        with rasterio.open(img_fp, 'w', **meta) as dst:
            # Update # description # for # the # stack
            dst.descriptions = srfiles_tags
            for _id, layer in enumerate(srfiles, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(_id, src1.read(1))
        return Rastermsp(img_fp)


    def load_rgb(self, height=None, width=None, col_off=0,row_off=0):
        if self.desc:
            try:
                # Use rasterio indexing
                red_idx = self.desc.index('Sent2_B04')+1
                green_idx = self.desc.index('Sent2_B03')+1
                blue_idx = self.desc.index('Sent2_B02')+1
            except ValueError:
                print("Band description of the form {} expected. Your desc is {} instead".format('Sent2_B0*',self.desc))
        else:
            raise ValueError("No available description for this raster")

        if width is None:
            width = self.width
        if height is None:
            height = self.height

        rgb = self.load_as_arr(height, width, [red_idx,green_idx,blue_idx], col_off, row_off)

        rgb_norm = np.empty((height,width,3), dtype=np.float32)
        rgb_norm[:,:,0] = self.normalize(rgb[:,:,0])
        rgb_norm[:,:,1] = self.normalize(rgb[:,:,1])
        rgb_norm[:,:,2] = self.normalize(rgb[:,:,2])

        # normalized RGB natural color composite
        return rgb_norm

# to be change with Rasterhsp()
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

    def get_raster_window(self, width, height, col_off=0,row_off=0):
        """
        Load the raster as numpy array
        """

        with rasterio.open(self.path_to_raster) as dataset:
            img = dataset.read(window=Window(col_off, row_off, width, height))
        return img


    def get_raster_tiles(self, tile_size_x=50, tile_size_y=70):

        with rasterio.open(self.path_to_raster) as dataset:
            for col in range(0, self.width, tile_size_x):
                for row in range(0, self.height, tile_size_y):
                    #print(row,col)
                    tile = dataset.read(window=Window(col, row, tile_size_x, tile_size_y))
                    yield tile.shape

if __name__ == '__main__':
    basedir = '/home/diego/work/dev/ess_diego/'

    imgdir = os.path.join(basedir,'Data_Diego/S2B_MSIL1C_20190812T073619_N0208_R092_T37MBN_20190812T102512.SAFE/GRANULE/L1C_T37MBN_A012702_20190812T075555/IMG_DATA')

    # Load raster stack
    #r = Rastermsp((os.path.join(imgdir,'Sentinel2_20170718_.tif')))
    # Load rsater ndvi
    r = Rastermsp((os.path.join(imgdir,'Sentinel2_20190812_.tif')))
    rgb_arr = r.load_RGB_layers()
