import os
import copy
import glob
import re
#from parser import Sentinel2
import rasterio
import rasterio.plot
from rasterio.plot import reshape_as_image, plotting_extent
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, mapping


class Raster:
    """
    Base raster class to process geo-referenced raster data.
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
        Load a raster image as a 3D numpy array. The axis of the array have the order:

        (height, width, bands) or
        (rows, cols, channels)

        This is the order expected by image processing and visualization software;
        i.e. matplotlib, scikit-image, etc..

        Use 'heigh' and 'width' to load only a window of the raster.
        If raster is a multi layer stack use 'band' to select a single or multiple bands of choice.
        ************

        params:
            height -> height (number of rows)
            width -> width (number of cols)
            bands -> number of layers
            col_off -> starting column
            row_off -> starting row
        """
        # check whether to load the entire array
        if height is None:
            height = self.height
        if width is None:
            width = self.width
        # check how many bands to load
        if bands is None:
            bands = [i for i in range(1, self.count+1)]
        else:
            # make sure bands is a list of indexes
            if not isinstance(bands, list):
                bands = [bands]

        with rasterio.open(self.path_to_raster) as dataset:
            img = dataset.read(bands, window=Window(col_off, row_off, width, height))
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

    def extract_pixel_value(self, gdf):
        """
        It returns pixel values at Point coordinates contained in gdf

        args:
            gdf -- Geopandas dataframe. A column named 'geometry' is expected
        """
        gdf_copy = gdf.copy()

        geoms = gdf_copy.geometry.values

        with rasterio.open(self.path_to_raster) as src:
            pixel_values = np.array([]).reshape(0,src.count)
            for index, geom in enumerate(geoms):
                feature = [mapping(geom)]
                #3D np array
                out_image, out_transform = mask(src, feature, crop=True)
                #2D np array
                out_image_reshaped = out_image.reshape(-1, src.count)
                pixel_values = np.vstack((pixel_values,out_image_reshaped))
        return pixel_values

    def get_raster_polygon(self):
        """
        It returns a Polygon Shapely object of the entire raster using
        witdh and height
        """

        endrow = self.height
        endcol = self.width

        top_left = self.transform_to_coordinates(0,0)
        bottom_left = self.transform_to_coordinates(endrow,0)
        top_right = self.transform_to_coordinates(0,endcol)
        bottom_right = self.transform_to_coordinates(endrow,endcol)
        coords = [top_left, bottom_left, bottom_right, top_right]

        return Polygon(coords)

    def get_gdf_within(self, gdf):
        """
        Return a new geodataframe with Points only within the
        Raster polygon
        """
        poly = self.get_raster_polygon()
        # create a mask
        mask = gdf.within(poly)
        gdf_within = gdf.loc[mask]
        return gdf_within

#    def write_tiles(self, tile_size_x=50, tile_size_y=70): # To be finished
#        with rasterio.open(self.path_to_raster) as dataset:
#            for col in range(0, self.width, tile_size_x):
#                for row in range(0, self.height, tile_size_y):
#                    #print(row,col)
#                    tile = dataset.read(window=Window(col, row, tile_size_x, tile_size_y))
#                    yield tile.shape

    @classmethod
    def mask_arr(cls, arr, mask):
        """
        Mask an array using a mask array.

        *********

        params:
            arr -> 3D numpy array (rows, cols, single channel)
            mask -> 3D boolean masked array

        """
        return np.ma.array(arr, mask=mask)

    @classmethod
    def mask_arr_greater_equal(cls, arr, val):
        """
        Mask the values of an array that are greater than or equal to a given threshold.

        *********

        params:
            arr -> 3D numpy array (rows, cols, single channel)
            val -> a threshold value

        """
        return np.ma.masked_greater_equal(arr, val)

    @classmethod
    def mask_arr_less_equal(cls, arr, val):
        """
        Mask the values of an array that are less than or equal to a given threshold.

        *********

        params:
            arr -> 3D numpy array (rows, cols, single channel)
            val -> a threshold value

        """
        return np.ma.masked_less_equal(arr, val)

    @classmethod
    def load_rgb(cls, red_obj, green_obj, blue_obj, height, width, bands=1, col_off=0,row_off=0):
        """
        Calculate and load an RGB 3D numpy array.

        *********

        params:
            red_obj, green_obj, blue_obj -> Instances of the Raster class
        """
        r = red_obj.load_as_arr(height, width, bands, col_off, row_off)
        g = green_obj.load_as_arr(height, width, bands, col_off, row_off)
        b = blue_obj.load_as_arr(height, width, bands, col_off, row_off)


        rgb_norm = np.empty((height,width,3), dtype=np.float32)
        rgb_norm[:,:,0] = Raster._normalize(r[:,:,0])
        rgb_norm[:,:,1] = Raster._normalize(g[:,:,0])
        rgb_norm[:,:,2] = Raster._normalize(b[:,:,0])

        # normalized RGB natural color composite
        return rgb_norm

    @classmethod
    def points_on_layer_plot(self, r_obj, layer_arr, gdf, **kwargs):

        layer_endrow = layer_arr.shape[0]
        layer_endcol = layer_arr.shape[1]
        layer_poly = [r_obj.transform_to_coordinates(0,0), r_obj.transform_to_coordinates(layer_endrow,0),
                      r_obj.transform_to_coordinates(layer_endrow,layer_endcol), r_obj.transform_to_coordinates(0,layer_endcol)]

        # check for raster and arr coordinates
        assert r_obj.get_raster_polygon() == Polygon(layer_poly), "Input array and raster must have same coordinates"

        cmap, marker, markersize, color, label = kwargs.get('r_cmap',"pink"), \
                                  kwargs.get('s_marker',"s"), \
                                  kwargs.get('s_markersize',30), \
                                  kwargs.get('s_color',"purple"), \
                                  kwargs.get('s_label',"classname")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(layer_arr[:,:,0],
                      # Set the spatial extent or else the data will not line up with your geopandas layer
                      extent=plotting_extent(r_obj),
                      cmap=cmap)
        gdf.plot(ax=ax,
                     marker=marker,
                     markersize=markersize,
                     color=color,
                     label=label)
        return ax

    @classmethod
    def calc_ndvi(cls, red_obj, nir_obj, height, width, bands=1, col_off=0, row_off=0, write=False):
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
    def _normalize(arr):
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
        rgb_norm[:,:,0] = self._normalize(rgb[:,:,0])
        rgb_norm[:,:,1] = self._normalize(rgb[:,:,1])
        rgb_norm[:,:,2] = self._normalize(rgb[:,:,2])

        # normalized RGB natural color composite
        return rgb_norm

    @classmethod
    def points_on_layer_plot(self, r_obj, arr, gdf, **kwargs):

        layer_arr = arr[:,:,0]
        layer_endrow = layer_arr.shape[0]
        layer_endcol = layer_arr.shape[1]
        layer_poly = [r_obj.transform_to_coordinates(0,0), r_obj.transform_to_coordinates(layer_endrow,0),
                      r_obj.transform_to_coordinates(layer_endrow,layer_endcol), r_obj.transform_to_coordinates(0,layer_endcol)]

        # check for raster and arr coordinates
        assert r_obj.get_raster_polygon() == Polygon(layer_poly), "Input array and raster must have same coordinates"

        cmap, marker, markersize, color, label = kwargs.get('r_cmap',"pink"), \
                                  kwargs.get('s_marker',"s"), \
                                  kwargs.get('s_markersize',30), \
                                  kwargs.get('s_color',"purple"), \
                                  kwargs.get('s_label',"classname")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(layer_arr[:,:,0],
                      # Set the spatial extent or else the data will not line up with your geopandas layer
                      extent=plotting_extent(r_obj),
                      cmap=cmap)
        gdf.plot(ax=ax,
                     marker=marker,
                     markersize=markersize,
                     color=color,
                     label=label)
        return ax

    def show_hist(self):
        with rasterio.open(self.path_to_raster) as dataset:
            rasterio.plot.show_hist(dataset.read([1,2,3,4]), bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)

class Rasterhsp(Raster):

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

    def calc_ndvi(self, red_arr, nir_arr, bands=1, col_off=0, row_off=0, write=False):
        """
        Calculate ndvi and write to disk if write=True

        ***********
        params:
            red_arr -> 3D numpy array
            nir_arr -> 3D numpy array

        The axis of the array have the order:

        (height, width, bands) or
        (rows, cols, channels)

        This is the order expected by image processing and visualization software;
        i.e. matplotlib, scikit-image, etc..

        ***********
        return:
            3D numpy array (ndvi)

        """
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_arr = (nir_arr.astype(float)-red_arr.astype(float))/(nir_arr.astype(float)+red_arr.astype(float))

        # write to disk
        if write:
            # grab and copy metadata of one of the two array
            ndvi_meta = copy.deepcopy(self.meta)
            ndvi_meta.update(count=1, dtype="float64", driver='GTiff')
            ndvi_path = os.path.dirname(self.path_to_raster)
            with rasterio.open(os.path.join(ndvi_path,'ndvi.gtif'), 'w', **ndvi_meta) as dst:
                dst.descriptions = ['ndvi']
                dst.write(ndvi_arr, 1)
        return ndvi_arr


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

if __name__ == '__main__':
    basedir = '/home/diego/work/dev/ess_diego/'

    imgdir = os.path.join(basedir,'Data_Diego/S2B_MSIL1C_20190812T073619_N0208_R092_T37MBN_20190812T102512.SAFE/GRANULE/L1C_T37MBN_A012702_20190812T075555/IMG_DATA')

    # Load raster stack
    #r = Rastermsp((os.path.join(imgdir,'Sentinel2_20170718_.tif')))
    # Load rsater ndvi
    r = Rastermsp((os.path.join(imgdir,'Sentinel2_20190812_.tif')))
    rgb_arr = r.load_RGB_layers()
