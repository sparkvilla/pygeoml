import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import copy
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from rasterio.plot import reshape_as_image
from pygeoml.raster import Raster, Rastermsp, Rasterhsp
from pygeoml.parser import Sentinel2
from pygeoml.shape import Shapeobj
from pygeoml.train import Trainingdata
from pygeoml.utils import raster_to_disk

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('../../pygeoml.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

class_to_int_map = [(b'Agricult', 1),
                    (b'Bare_roc', 2),
                    (b'Building', 3),
                    (b'Charcoal', 4),
                    (b'Forest', 5),
                    (b'Livestoc', 6),
                    (b'Marsh', 7),
                    (b'River', 8),
                    (b'Road', 9),
                    (b'Tree_pla', 10),
                    (b'masked', 11)]

def _to_csv(fpath, ci_map, n_est, acc):
    with open(fpath, 'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['class','num'])
        for row in data:
            csv_out.writerow(row)
        csv_out.writerow(['n_estimators','accuracy'])
        csv_out.writerow([n_est, acc])

def str_class_to_int(class_array):
    # Transform string class in integer class
    class_array[class_array == class_to_int_map[0][0]] = class_to_int_map[0][1]
    class_array[class_array == class_to_int_map[1][0]] = class_to_int_map[1][1]
    class_array[class_array == class_to_int_map[2][0]] = class_to_int_map[2][1]
    class_array[class_array == class_to_int_map[3][0]] = class_to_int_map[3][1]
    class_array[class_array == class_to_int_map[4][0]] = class_to_int_map[4][1]
    class_array[class_array == class_to_int_map[5][0]] = class_to_int_map[5][1]
    class_array[class_array == class_to_int_map[6][0]] = class_to_int_map[6][1]
    class_array[class_array == class_to_int_map[7][0]] = class_to_int_map[7][1]
    class_array[class_array == class_to_int_map[8][0]] = class_to_int_map[8][1]
    class_array[class_array == class_to_int_map[9][0]] = class_to_int_map[9][1]
    class_array[class_array == class_to_int_map[10][0]] = class_to_int_map[10][1]
    return(class_array.astype('uint8'))


def step1(dir10, dir20, fpath_scl, outdir):
    """
    Create a cloud masked stack array
    """
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

    p = Sentinel2(dir10)
    # get raster files
    rfiles = [ i[0] for i in p.srfiles]

    # 2. create the stack
    r_stack = Rastermsp.create_stack(rfiles, outdir, arr_scf_resampled_mask.mask)

def step2(dirshp, outdir):
    """
    Extract pixels at point location and create training data
    """
    stackdir = os.path.join(outdir,'multibands_masked.gtif')
    stack = Rastermsp(stackdir)
    pt = Shapeobj(dirshp)
    train = Trainingdata.calc_xy(stack, pt.gdf, write=True, outdir=outdir)
    train.add_class_xy("masked", 0)
    # Esclude classes with samples less then 10
    train.exclude_classes()
    train.save_xy(stack.path_to_raster, outdir)

def step3(outdir):
    """
    Use k-fold cross validation to get the best n_estimator param based on accuracy
    """
    train = Trainingdata.load_xy(os.path.join(outdir,'multibands_masked_features.npy'), os.path.join(outdir,'multibands_masked_lables.npy'))

    # Choose the best n_estimator value using k-fold cross validation
    n_est_range = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    n_est_scores = []

    for n in n_est_range:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True)
        scores = cross_val_score(rf, train.X, train.y, cv=10, scoring='accuracy')
        n_est_scores.append(scores.mean())

    best_score = max(n_est_scores)
    logger.info('Best score found: {}'.format(best_score) )

    # get index of the maximum score
    idx_max = n_est_scores.index(best_score)

    best_estimator = n_est_range[idx_max]
    logger.info('Best estimator: {}'.format(best_estimator) )
    return best_estimator, best_score

def step4(outdir, n_estimator):
    """
    Make prediction on real data using a pre-optimized classifier
    """
    stackdir = os.path.join(outdir,'multibands_masked.gtif')
    stack = Rastermsp(stackdir)
    train = Trainingdata.load_xy(os.path.join(outdir,'multibands_masked_features.npy'), os.path.join(outdir,'multibands_masked_lables.npy'))
    #  Initialize our model with n_estimator trees
    rf_f = RandomForestClassifier(n_estimators=n_estimator, oob_score=True)
    # Fit our model to training data
    rf_f = rf_f.fit(train.X, train.y)
    logger.info('Classes: {}'.format(np.unique(train.y)))
    class_prediction = Trainingdata.predict(stack, '|S8', 3, 3, rf_f, write=True, outdir=outdir)

def step5(outdir):
    """
    Convert prediction map back to raster
    """
    stackdir = os.path.join(outdir,'multibands_masked.gtif')
    stack = Rastermsp(stackdir)
    new_meta = stack.meta
    new_meta.update(count = 1)
    new_meta.update(dtype = 'uint8')
    new_meta.update(driver = 'GTiff')

    class_prediction = np.load(os.path.join(outdir,'multibands_masked_class_prediction.npy'))
    class_prediction = str_class_to_int(class_prediction)

    # add a third axes (channel) to the np array 
    class_prediction = np.expand_dims(class_prediction, axis=2)
    logger.info('class prediction int'.format(np.unique(class_prediction)))
    raster_to_disk(class_prediction, 'scene_classification', new_meta, outdir)
    
    # add map file
    #fpath = os.path.join(outdir, 'scene_classification_map.csv')
    #_to_csv(fpath, class_to_int_map)

def plot_class_prediction(outdir):

    # find the highest pixel value in the prediction image
    n = int(np.max(class_prediction))
    # next setup a colormap for our map
    colors = dict((
        (0, (218, 165, 32)),   # golden rod - Agri
        (1, (188, 143, 143)),  # rosy brown - Rock
        (2, (119, 136, 153)),  # light slate gray - Building
        (3, (139, 69, 19)),    # saddle brown  - Charcol
        (4, (34, 139, 34)),    # forest green - forest
        (5, (255, 255, 0)),    # yellow - livestock
        (6, (0, 0, 128)),    # navy - marsh
        (7, (127, 255, 212)),  # acqua marine - river
        (8, (178, 34, 34)),    # firebrick - road
        (9, (0, 255, 0)),    # lime - tree_plantation
        (10, (255, 255, 255)),  # White - masked
    ))

    # Transform 0 - 255 color values from colors as float 0 - 1
    for k in colors:
        v = colors[k]
        _v = [_v / 255.0 for _v in v]
        colors[k] = _v
    index_colors = [colors[key] for key in range(0, n )]

    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n)


    from matplotlib.patches import Patch
    # Create a list of labels to use for your legend
    class_labels = ['Agricultur', 'Bare_rock', 'Building', 'CharcoalKi', 'Forest',
           'Livestock', 'Marsh', 'River', 'Road', 'Tree_plant', 'masked']
    # A path is an object drawn by matplotlib. In this case a patch is a box draw on your legend
    # Below you create a unique path or box with a unique color - one for each of the labels above
    legend_patches = [Patch(color=icolor, label=label)
                      for icolor, label in zip(index_colors, class_labels)]

    # Plot Classification
    fig, axs = plt.subplots(1,1,figsize=(30,30))
    axs.imshow(class_prediction, cmap=cmap, interpolation='none')
    axs.legend(handles=legend_patches,
              facecolor="white",
              edgecolor="white",
              bbox_to_anchor=(1.35, 1),
              fontsize=30)  # Place legend to the RIGHT of the map
    axs.set_axis_off()
    plt.show()




if __name__ == "__main__":

    basedir = '/mnt/outdata/Hanneke/ESSCharcoal_1_07to112019Sentinel_1_out'

    # output base directory
    base_outdir = os.path.join(basedir,'output')

    # shape files
    datadir_shp = os.path.join(basedir,'field_points_dataframe')

    data_paths = ['S2A_MSIL2A_20190628T073621_N9999_R092_T37MBN_20191121T145522.SAFE/GRANULE/L2A_T37MBN_A020967_20190628T075427',
            'S2A_MSIL2A_20190906T073611_N0213_R092_T37MBN_20190906T110000.SAFE/GRANULE/L2A_T37MBN_A021968_20190906T075543',
            'S2A_MSIL2A_20190906T073611_N9999_R092_T37MBN_20191121T152023.SAFE/GRANULE/L2A_T37MBN_A021968_20190906T075543',
            'S2A_MSIL2A_20191026T074011_N9999_R092_T37MBN_20191121T154707.SAFE/GRANULE/L2A_T37MBN_A022683_20191026T075457',
            'S2B_MSIL2A_20190623T073619_N0212_R092_T37MBN_20190623T120816.SAFE/GRANULE/L2A_T37MBN_A011987_20190623T075723',
            'S2B_MSIL2A_20190623T073619_N9999_R092_T37MBN_20191121T161217.SAFE/GRANULE/L2A_T37MBN_A011987_20190623T075723',
            'S2B_MSIL2A_20190703T073619_N9999_R092_T37MBN_20191121T163720.SAFE/GRANULE/L2A_T37MBN_A012130_20190703T075901',
            'S2B_MSIL2A_20190713T073619_N9999_R092_T37MBN_20191121T170202.SAFE/GRANULE/L2A_T37MBN_A012273_20190713T075952',
            'S2B_MSIL2A_20190723T073619_N9999_R092_T37MBN_20191121T172709.SAFE/GRANULE/L2A_T37MBN_A012416_20190723T075433',
            'S2B_MSIL2A_20190812T073619_N0213_R092_T37MBN_20190812T105742.SAFE/GRANULE/L2A_T37MBN_A012702_20190812T075555',
            'S2B_MSIL2A_20190812T073619_N9999_R092_T37MBN_20191121T175134.SAFE/GRANULE/L2A_T37MBN_A012702_20190812T075555']

    for dp in data_paths:
        dirname = dp.split('/')[0]
        logger.info('Start workflow for {}'.format(dirname))
        outdir = os.path.join(base_outdir, dirname + '_out')
        # Create target Directory if don't exist
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            logger.info('Directory {} created'.format(outdir))
        else:
            logger.info('Directory {} already exists'.format(outdir))


        # 2A-Level original data 10 m
        datadir_10m = os.path.join(basedir, dp, 'IMG_DATA/R10m')
        # 2A-Level original data 20 m
        datadir_20m = os.path.join(basedir, dp, 'IMG_DATA/R20m')

        # get the SCL file path
        for scl in glob.glob(os.path.join(datadir_20m, '*SCL*.jp2')):
                fpath_scl = scl

        #step1(datadir_10m, datadir_20m, fpath_scl, outdir)
        #step2(datadir_shp, outdir)
        #best_est = step3(outdir)
        #step4(outdir, best_est)
        step5(outdir)
        logger.info('Finished workflow for {}'.format(dirname))
