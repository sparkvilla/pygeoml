import os
import glob
import copy

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

    basedir = '/home/diego/work/dev/ess_diego/'

    imgdir = os.path.join(basedir,'Data_Diego/Sentinel/Sentinel_2017/S2B_MSIL1C_20171018T072859_N0205_R049_T37MBN_20171018T074738.SAFE/GRANULE/L1C_T37MBN_A003221_20171018T074738/IMG_DATA/')
    #r = Raster(os.path.join(imgdir,'T37MBN_20170718T075211_B02.jp2'))
    #r = Rastermsp.merge_to_single_raster(imgdir,Sentinel2)
    r = Rastermsp.merge_to_single_raster(imgdir,Sentinel2)
    print(r.desc)
