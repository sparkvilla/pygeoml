import matplotlib.pyplot as plt
import os
import glob
import copy

from rasterio.plot import reshape_as_image
from raster import Raster, Rastermsp
from parser import Sentinel2
from shape import Shapeobj
from train import Trainingdata

def calc_ndvi(path_to_raster, outdir):
    raster = Rasterobj(path_to_raster, outdir)
    raster.get_red_and_nir_bands()
    raster.get_ndvi_band(write=True)

def build_geo_df(dirpath_to_shapes):
    sfiles = [sfile for sfile in glob.glob(os.path.join(dirpath_to_shapes, '*.shp'))]
    shapeobjs = [Shapeobj(sfile) for sfile in sfiles ]
    for so in shapeobjs:
        so.drop_columns('style_css', 'label', 'text', 'dateTime')
        so.reproject('epsg:26912')
    return [sh.gdf for sh in shapeobjs]

def wf1():
    inpRaster = os.path.join(basedir,'Yuma','Finalsubset.dat')
    inpShapes = os.path.join(basedir,'Shapefiles_accurate')
    outdir = os.path.join(basedir,'output')

    calc_ndvi(inpRaster, outdir)
    shps_list = build_geo_df(inpShapes)

    td1 = Trainingdata.calc_xy(inpRaster, shps_list)
    td1.add_background_xy()
    td1.save_xy('base_extracted',outdir)

    td2 = Trainingdata.calc_xy(os.path.join(outdir,'ndvi.gtif'), shps_list)
    td2.add_background_xy()
    td2.save_xy('ndvi_extracted',outdir)

    td3 = td1.hstack_to_x(td2)
    td3.save_xy('base_and_ndvi_extracted',outdir)

if __name__ == "__main__":

    basedir = '/home/diego/work/dev/ess_diego/Data_Diego'


    #Get data with 10m resolution
    datadir = os.path.join(basedir,
            'S2B_MSIL2A_20190812T073619_N9999_R092_T37MBN_20190919T144441.SAFE/GRANULE/L2A_T37MBN_A012702_20190812T075555/IMG_DATA/R10m/')
    shpdir = os.path.join(datadir,'field_points')
    fpath_r = os.path.join(datadir,'multibands_20190812.gtif')
    fpath_ndvi = os.path.join(datadir,'ndvi.gtif')
    fpath_red = os.path.join(datadir,'T37MBN_20190812T073619_B04_10m.jp2')
    fpath_shape = os.path.join(shpdir,'field_points.shp')
    # Instanciate stack, red, ndvi and shape objects
    r = Rastermsp(fpath_r)
    r_ndvi = Raster(fpath_ndvi)
    r_red = Raster(fpath_red)
    shapes = Shapeobj(fpath_shape)
    red_arr = r_red.load_as_arr()

    # load a new gdf that excludes points outside the raster polygon boundaries
    new_gdf = r_red.get_gdf_within(shapes.gdf)

    Raster.points_on_layer_plot(r_red, red_arr, new_gdf, s_markersize=7)

    plt.show()
