import os
import glob
import copy

from raster import Rasterobj
from shape import Shapeobj
from train import Trainingdata

def calc_ndvi(path_to_raster, outdir):
    raster = Rasterobj(path_to_raster, outdir)
    raster.get_red_and_nir_bands()
    raster.get_ndvi_band(write=True)

def build_geo_df(dirpath_to_shapes):
    sfiles = [file for file in glob.glob(os.path.join(dirpath_to_shapes, '*.shp'))]
    shapeobjs = [Shapeobj(sfile) for sfile in sfiles ]
    for so in shapeobjs:
        so.drop_columns('style_css', 'label', 'text', 'dateTime')
        so.reproject('epsg:26912')
    return [sh.gdf for sh in shapeobjs]

def build_features_lables(path_to_raster, list_of_gdfs, outdir):
    a = Trainingdata.calc_xy(path_to_raster, *list_of_gdfs)
    a.add_background_xy()
    a.save_xy('Finalsubset',outdir)
    return a

if __name__ == "__main__":

    basedir = '/home/diego/work/dev/ess_diego/'

    inpRaster = os.path.join(basedir,'Yuma','Finalsubset.dat')
    inpShapes = os.path.join(basedir,'Shapefiles_accurate')
    outdir = os.path.join(basedir,'output')

    calc_ndvi(inpRaster, outdir)
    shps_list = build_geo_df(inpShapes)
    r1 = build_features_lables(inpRaster, shps_list, outdir)
    r2 = build_features_lables(os.path.join(outdir,'ndvi.gtif'), shps_list, outdir)
    r3 = r1.hstack_to_x(r2)
    r3.save_xy('Finalxy',outdir)
