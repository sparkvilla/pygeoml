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
    imgdir = os.path.join(basedir,
            'S2B_MSIL2A_20190812T073619_N9999_R092_T37MBN_20190919T144441.SAFE/GRANULE/L2A_T37MBN_A012702_20190812T075555/IMG_DATA/R10m/')


    #stack = Rastermsp.create_stack(os.path.join(basedir,imgdir),Sentinel2)

    green_path = os.path.join(imgdir, 'T37MBN_20190812T073619_B03_10m.jp2')
    blue_path = os.path.join(imgdir, 'T37MBN_20190812T073619_B02_10m.jp2')
    red_path = os.path.join(imgdir, 'T37MBN_20190812T073619_B04_10m.jp2')
    nir_path = os.path.join(imgdir, 'T37MBN_20190812T073619_B08_10m.jp2')
    red = Raster(red_path)
    blue = Raster(blue_path)
    green = Raster(green_path)
    nir = Raster(nir_path)

    ndvi = Raster.load_ndvi(red, nir, width=red.width, height=red.height, write=True)
    #rgb = Raster.load_rgb(red,green,blue,height=500,width=500,col_off=0,row_off=0)
    #stack_filepath = os.path.join(imgdir,'Sentinel2_20190812_.tif')
    #stack = Rastermsp(stack_filepath)
    #ndvi = r_ndvi.load_window(width=500, height=500, col_off=8000,row_off=10000)
    #grass = r_ndvi.get_pixel_value(56+10000,398+8000)
    #cloud = r_ndvi.get_pixel_value(10+10000,485+8000)
    #sand = r_ndvi.get_pixel_value(56+10000,419+8000)
    #print("Green: {}".format(grass))
    #print("Sand: {}".format(sand))
    #print("Cloud: {}".format(cloud))
    #rgb = stack.load_rgb(height=500,width=500,col_off=0,row_off=0)
    #plt.imshow(rgb)
    #fig, ax = plt.subplots(figsize=(10, 10))
    #ax.imshow(rgb)
    #plt.show()
