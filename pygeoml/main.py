import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import copy
import sys

from rasterio.plot import reshape_as_image
from pygeoml.raster import Raster, Rasterhsp
from pygeoml.parser import Sentinel2
from pygeoml.shape import Shapeobj
from pygeoml.train import Trainingdata
from pygeoml.utils import raster_to_disk, load_json

import logging
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def step0(basedir):
    "Select tiles with less then 30% masking"

    logger.info('Start step0..{}'.format(step0.__doc__))
    final_dirs = []

    dirs_20m = [x[0] for x in os.walk(basedir) if 'GRANULE' and 'IMG_DATA' and 'R20m' in x[0]]
    fpaths_20m = [glob.glob(os.path.join(dirpath, '*SCL*.jp2'))[0] for dirpath in dirs_20m]

    from pathlib import Path
    dirs = [ str(Path(fpath).parents[2]) for fpath in fpaths_20m]

    for fpath_20m, _dir in zip(fpaths_20m, dirs):
        # 1. create a cloud mask at 10m resol based on scf file
        r_scf = Raster(fpath_20m)
        ## Load arr_scf in memory
        arr_scf = r_scf.load_as_arr()
        # classification_mask:
        # 3 -> cloud_shadow
        # 8 -> cloud_medium_probability
        # 9 -> cloud_high_probability
        # see https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
        arr_scf_mask = Raster.mask_arr_equal(arr_scf, [3,7,8,9,10])
        not_masked = arr_scf_mask.count()
        masked = np.ma.count_masked(arr_scf_mask)
        perc = masked / (masked+not_masked)

        if perc <= 0.40:
            logger.info('Added: {}, Masked portion: {}'.format(_dir, perc))
            final_dirs.append((_dir, perc))

    return final_dirs

def step1(dir10, dir20, fpath_scl, outdir):
    "Create a cloud masked stack array"

    logger.info('Start step1..{}'.format(step1.__doc__))
    # 1. create a cloud mask at 10m resol based on scf file
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

    # create Sentinel2 parser
    p = Sentinel2(dir10)

    r_obj_path = [i[0] for i in p.fpaths if i[1] == 'Sent2_B04'][0]
    nir_obj_path = [i[0] for i in p.fpaths if i[1] == 'Sent2_B08'][0]

    r_obj = Raster(r_obj_path)
    nir_obj = Raster(nir_obj_path)

    ndvi_path = Raster.calc_ndvi(r_obj, nir_obj, outdir)
    # get raster files
    rfiles = [ i[0] for i in p.sfpaths]
    rfiles.append(ndvi_path.path_to_raster)

    # 2. create the stack
    r_stack = Raster.create_stack(rfiles, outdir, 'float32', arr_scf_resampled_mask.mask)

def step2(dirshp, outdir):
    "Use the stack to create a prediction map"

    logger.info('Start step2..{}'.format(step2.__doc__))
    # Extract pixels at point location and create training data
    stackdir = os.path.join(outdir,'multibands_masked.gtif')
    stack = Raster(stackdir)
    pt = Shapeobj(dirshp)
    train = Trainingdata.calc_xy(stack, pt.gdf)
    train.add_class("masked", 0)
    # Esclude classes with samples less then 10
    train.exclude_classes()
    train.save(outdir)

    # Use Random forest classifier
    estimator, accuracy = train.random_forest_cross_val()
    rf_cl = train.random_forest_train(estimator)
    class_prediction = train.predict(stack, 3, 3, rf_cl, write=True, outdir=outdir)
    # convert to uint8
    class_prediction = class_prediction.astype(np.uint8)

    # Convert prediction map back to raster
    new_meta = stack.meta
    new_meta.update(count = 1)
    new_meta.update(dtype = 'uint8')
    new_meta.update(driver = 'GTiff')
    # add a third axes (channel) to the np array
    class_prediction = np.expand_dims(class_prediction, axis=2)
    raster_to_disk(class_prediction, 'scene_classification', new_meta, outdir)


if __name__ == "__main__":


    basedir = '/mnt/outdata/Hanneke/ESSCharcoal_from201908_to202002'
    # output base directory
    base_outdir = os.path.join(basedir,'output_20200324')
    data_paths = step0(basedir)
    # shape files
    datadir_shp = os.path.join(basedir,'all_points')

    for dp in data_paths:
        dirname = dp[0].split('/')[-3]
        logger.info('Start workflow for {}'.format(dirname))
        logger.info('Cloud coverage {}'.format(dp[1]))
        outdir = os.path.join(base_outdir, dirname + '_out')
        # Create target Directory if don't exist
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            logger.info('Directory {} created'.format(outdir))
        else:
            logger.info('Directory {} already exists'.format(outdir))


        # 2A-Level original data 10 m
        datadir_10m = os.path.join(dp[0], 'IMG_DATA/R10m')
        # 2A-Level original data 20 m
        datadir_20m = os.path.join(dp[0], 'IMG_DATA/R20m')

        # get the SCL file path
        fpath_scl = glob.glob(os.path.join(datadir_20m, '*SCL*.jp2'))[0]

        step1(datadir_10m, datadir_20m, fpath_scl, outdir)
        step2(datadir_shp, outdir)
        logger.info('Finished workflow for {}'.format(dirname))
        logger.info('\n')
