import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import copy
from sklearn.ensemble import RandomForestClassifier

from rasterio.plot import reshape_as_image
from raster import Raster, Rastermsp, Rasterhsp
from parser import Sentinel2
from shape import Shapeobj
from train import Trainingdata


if __name__ == "__main__":


    basedir = '/home/diego/work/dev/ess_diego/Data_Diego'

    # 2A-Level original data 10 m
    datadir_10m = os.path.join(basedir,'Hanneke/S2A_MSIL2A_20191026T074011_N9999_R092_T37MBN_20191121T154707.SAFE/GRANULE/L2A_T37MBN_A022683_20191026T075457/IMG_DATA/R10m')
    # 2A-Level original data 20 m
    datadir_20m = os.path.join(basedir,'Hanneke/S2A_MSIL2A_20191026T074011_N9999_R092_T37MBN_20191121T154707.SAFE/GRANULE/L2A_T37MBN_A022683_20191026T075457/IMG_DATA/R20m')
    # shape files
    datadir_shp = os.path.join(basedir,'Hanneke/S2A_MSIL2A_20191026T074011_N9999_R092_T37MBN_20191121T154707.SAFE/field_points_dataframe')

    # output directory
    outdir = os.path.join(basedir,'Hanneke/output/S2A_MSIL2A_20191026T074011_N9999_R092_T37MBN_20191121T154707.SAFE_out')

    # 1. create a cloud mask at 10m resol based on scf file
    fpath_scl = os.path.join(datadir_20m,'T37MBN_20191026T074011_SCL_20m.jp2')
    r_scf = Raster(fpath_scl)
    ## Upsample scene classification Raster image
    r_scf_resampled = Raster.resample_raster(r_scf, scale=2, outdir=outdir)
    ## Load arr_scf_resampled in memory
    arr_scf_resampled = r_scf_resampled.load_as_arr()

    # classification_mask:
    # 3 -> cloud_shadow
    # 8 -> cloud_medium_probability
    # 9 -> cloud_high_probability
    # see https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    arr_scf_resampled_mask = Raster.mask_arr_equal(arr_scf_resampled, [3,7,8,9,10])
    arr_scf_resampled_mask.dump(os.path.join(outdir,'mask_10m.npy'))

    # 2. create the stack
    r_stack = Rastermsp.create_stack(datadir_10m, Sentinel2, outdir)
    arr_stack = r_stack.load_as_arr()
    # 3. create a masked stack
    arr_stack_masked = r_stack.mask_arr(arr_stack, arr_scf_resampled_mask.mask, write=True, outdir=outdir)





    # Load raster
#    fpath_red = os.path.join(datadir,'T37MBN_20190628T073621_B04_10m.jp2')
#    r_red = Raster(fpath_red)
#    poly = r_red.get_raster_polygon()

    # Field measurments
    #shpdir = os.path.join(outdir,'field_points_dataframe')
   # stackdir = os.path.join(outdir,'multibands_masked.gtif')
    #pt = Shapeobj(os.path.join(shpdir,'field_points_dataframe.shp'))
    #field_pt.gdf_within_polygon(poly)
    #field_pt.rename_field("Tree plantation", "Tree_plantation")
    #field_pt.rename_field("Bare rock", "Bare_rock")
    #field_pt.write_gdf(outdir=outdir)

    #stack = Rastermsp(stackdir)

    # get a mask for this dataset
    #c_mask = np.load(os.path.join(outdir,'mask_10m'), allow_pickle=True)
    # Mask stack
    #stack_masked = stack.mask_arr(stack_arr, c_mask.mask, write=True, outdir=outdir)
#    train = Trainingdata.calc_xy(stackdir, pt.gdf, write=True, outdir=outdir)
   # train = Trainingdata.load_xy(os.path.join(outdir,'multibands_masked_features.npy'), os.path.join(outdir,'multibands_masked_lables.npy'))
   # train.X = np.delete(train.X, 304, axis=0)
   # train.y = np.delete(train.y, 304 )
   # train.build_class_to_int_map()

# Initialize our model with 400 trees
    #rf_f = RandomForestClassifier(n_estimators=400, oob_score=True)
    # Fit our model to training data
    #rf_f = rf_f.fit(train.X, train.y)
    ####class_prediction = Trainingdata.predict(stack, 3, 3, rf_f, write=True, outdir=outdir)
