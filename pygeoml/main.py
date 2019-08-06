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

if __name__ == "__main__":

    basedir = '/home/diego/work/dev/ess_diego/'

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
