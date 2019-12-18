import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import copy

from rasterio.plot import reshape_as_image
from raster import Raster, Rastermsp, Rasterhsp
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
    #basedir = '/home/diego/work/dev/ess_diego/Data_Diego'

    #outdir = '/home/diego/work/dev/ess_diego/Data_Diego/Oli/output'
    #fpathr_g_st_l = os.path.join(outdir, 'f020414t02p01r06_geo_s01_georeferenced_stitched.gtif')
    #fpathr_g_st_r = os.path.join(outdir, 'f020414t02p01r07_geo_s01_georeferenced_stitched.gtif')
    #mos, ot = Rasterhsp.merge_rasters([fpathr_g_st_l,fpathr_g_st_r])

    basedir = '/home/diego/work/dev/ess_diego/Data_Diego'

    # 2A-Level original data 10 m
    datadir = os.path.join(basedir,'Hanneke/S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427/IMG_DATA/R10m')
    outdir = os.path.join(basedir,'Hanneke/output/S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE')

    # Load raster
    fpath_red = os.path.join(datadir,'T37MBN_20190628T073621_B04_10m.jp2')
    r_red = Raster(fpath_red)
    poly = r_red.get_raster_polygon()

    # Field measurments
    shpdir = os.path.join(outdir,'field_points_dataframe')
    stackdir = os.path.join(outdir,'multibands_masked.gtif')
    pt = Shapeobj(os.path.join(shpdir,'field_points_dataframe.shp'))
    #field_pt.gdf_within_polygon(poly)
    #field_pt.rename_field("Tree plantation", "Tree_plantation")
    #field_pt.rename_field("Bare rock", "Bare_rock")
    #field_pt.write_gdf(outdir=outdir)

    stack = Rastermsp(stackdir)
    # get a mask for this dataset
    #c_mask = np.load(os.path.join(outdir,'mask_10m'), allow_pickle=True)
    stack_arr = stack.load_as_arr()
    # Mask stack
    #stack_masked = stack.mask_arr(stack_arr, c_mask.mask, write=True, outdir=outdir)
    train = Trainingdata.calc_xy(stackdir, pt.gdf, write=True, outdir=outdir)
