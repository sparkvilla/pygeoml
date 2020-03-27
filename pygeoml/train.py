import os
import json
import sys

import rasterio
import rasterio.plot
from rasterio.mask import mask
from rasterio.merge import merge
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from shapely.geometry import mapping
from pygeoml.utils import np_to_disk, NumpyEncoder, to_json, json_to_disk, load_json

import logging
import pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainingdata:
    """
    Generate/modify:
      X features
      y labels
    used as training data for various ML algorithms.
    """
    def __init__(self, X, y, map_cl_id=None):
        self.X = X
        self.y = y
        self.map_cl_id = map_cl_id

        self.label_ids, self.label_ids_count = np.unique(self.y, return_counts=True)


    def save(self, outdir):
        logger.debug('Writing training data to disk')
        json_arr = to_json({'X':self.X, 'y':self.y, 'map_cl_id':self.map_cl_id}, encoder=NumpyEncoder)
        json_to_disk(json_arr, 'training_data', outdir)


    def add_class(self, label_name, value):
        """
        Extend X,y in the 0 dimension by adding a number of
        values to X and a numbers of label_id to y.

        map_cl_id is updated with a new mapping: label_name,
        label_id

        """

        logger.debug('''Current Labels ids: {}
                     Sample numbers: {} '''.format(self.label_ids, self.label_ids_count))
        logger.debug('Adding a new class to the training data ...')

        # set number of samples equal to the highest number of existing samples
        samples_len = np.max(self.label_ids_count)

        # make a label_id for the new label_name
        label_id = max(self.label_ids) + 1

        logger.debug('''Adding a new class, Label name: "{}", Value: {},
                        Label id: {}, Samples: {}'''.format(label_name, value, label_id, samples_len))

        # create arrays of values and label_ids
        values = np.ones((samples_len, self.X.shape[1]))*value
        ids = np.array([label_id for _ in range(samples_len)])

        # update instance attributes
        self.X = np.vstack((self.X, values))
        self.y = np.append(self.y, ids)
        self.label_ids = np.append(self.label_ids, label_id)
        self.label_ids_count = np.append(self.label_ids_count, samples_len)
        self.map_cl_id[str(label_id)] = label_name

    def hstack_to_x(self, other):
        # check x for equal 0 dimension
        assert self.X.shape[0] == other.X.shape[0]
        # check y for equal labels
        np.testing.assert_array_equal(self.y,other.y)
        # create a final array
        final_x = np.hstack((self.X,other.X))
        return Trainingdata(final_x, self.y, self.mapping)

    def exclude_classes(self, n=10):
        """
        Esclude classes with samples less then a minimum number

        *************

        params:
            n -> minimum number of samples (default 10)
        """
        logger.debug('Current Labels ids: {} Sample numbers: {} '.format(self.label_ids, self.label_ids_count))

        # get label_lids and counts of less then a minumum number n
        frequence =  dict(zip(self.label_ids, self.label_ids_count))
        lowfrequence = list(dict(filter(lambda elem: elem[1] < n,
                                      frequence.items())).keys())

        if lowfrequence:
            logger.debug('Label_ids: {} will be removed'.format(lowfrequence))
            idx_mask_xy = np.isin(self.y, lowfrequence)
            self.y = self.y[~idx_mask_xy]
            self.X = self.X[~idx_mask_xy]

            # update label_ids, label_ids_count and map_cl_id
            idx_mask = np.isin(self.label_ids, lowfrequence)
            self.label_ids = self.label_ids[~idx_mask]
            self.label_ids_count = self.label_ids_count[~idx_mask]
            for label_id in lowfrequence:
                # delete entries with low frequence occurrencies
                del self.map_cl_id[str(label_id)]
        logger.debug('Try exclude function but number of samples per-class is greater than {}'.format(n))


    def random_forest_cross_val(self, cv=10):
        """
        Use k-fold cross validation to get the best n_estimator param based on accuracy
        """
        logger.debug('Starting random forest cross validation... ')
        logger.debug('Cross validation param: {}'.format(cv))
        # Choose the best n_estimator value using k-fold cross validation
        n_est_range = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        n_est_scores = []

        for n in n_est_range:
            rf = RandomForestClassifier(n_estimators=n, oob_score=True)
            scores = cross_val_score(rf, self.X, self.y, cv=cv, scoring='accuracy')
            logger.debug('Estimators: {} -> Accuracy: {}'.format(n, scores.mean()))
            n_est_scores.append(scores.mean())

        best_score = max(n_est_scores)
        logger.info('Best score found: {}'.format(best_score) )

        # get index of the maximum score
        idx_max = n_est_scores.index(best_score)

        best_estimator = n_est_range[idx_max]
        logger.info('Best estimator: {}'.format(best_estimator) )
        return best_estimator, best_score


    def random_forest_train(self, n_estimator):
        logger.debug('Start training random forest classifier with {} estimators'.format(n_estimator))
        # Initialize the model with n_estimator trees
        rf = RandomForestClassifier(n_estimators=n_estimator, oob_score=True)
        # Fit the  model to training data
        rf = rf.fit(self.X, self.y)
        return rf


    def predict(self, r_obj, v_split, h_split, classifier, write=False, outdir=None):
        """
        Use an already optimized skitlearn classifier to classify pixels of
        a raster object.

        ************

        params:
            r_obj -> An istance of the Raster class
            v_split -> number of times the raster will be divided
                       along its height
            h_split -> number of times the raster will be divided
                       along its width
            classifier -> optimised classifier
            outdir -> full path to output directory

        return:
            class_final: A 2D numpy array of classnames with
                         the same height and width of r_obj
        """
        # Allocate a numpy array for strings
        logger.debug('Start predictions on real data...')
        class_final = np.empty((r_obj.height,r_obj.width), dtype =np.int64)
        logger.debug('Numpy arr allocated for classname of type {}'.format(class_final.dtype))

        for row, col, patch in r_obj.get_patches(v_split, h_split):
            np.nan_to_num(patch, copy=False,nan=0.0, posinf=0,neginf=0)
            # Generate a prediction array
            # cols: a single patch reshaped into a vector
            # rows: bands of a single patch
            class_prediction = classifier.predict(patch.reshape(-1, patch.shape[2]))
            # Reshape our classification map back into a 2D matrix so we can visualize it
            class_prediction = class_prediction.reshape(patch[:,:,0].shape)
            # fill in the class_final with class names
            class_final[row:row+patch.shape[0],col:col+patch.shape[0]] = class_prediction
        if write:
            logger.debug('Writing class prediction to disk')
            json_arr = to_json({'class_prediction':class_final}, encoder=NumpyEncoder)
            json_to_disk(json_arr, 'class_prediction', outdir)
        return class_final


    @classmethod
    def to_df(cls, X, y):
        # check X feature and y labels have the same 0 dimension
        assert X.shape[0] == y.shape[0], "X and y should have the same 0 dimension"
        df = pd.DataFrame(data=X,
                          index=[i for i in range(0, X.shape[0])],
                          columns=["Band"+str(i) for i in range(1, X.shape[1]+1)])
        df['labels'] = y.tolist()

        # !! should return df[id] too !!
        return df

    @classmethod
    def load_xy(cls, path_to_training_data):
        data = load_json(path_to_training_data)
        X = np.asarray(data['X'])
        y = np.asarray(data['y'])
        map_cl_id = data['map_cl_id']
        return Trainingdata(X,y,map_cl_id)

    @classmethod
    def calc_xy(cls, r_obj, gdf, write=False, outdir=None):
        """
        Build the X feature matrix by extracting the pixel values from the raster
        at the point location.
        Build the y labels vector by extracting the class name from geo dataframes
        at the point location.

        Write X,y to disk at path_to_raster

        X         col1 col2 ... band_n          y
        Point1    val1 val2                     classname
        Point2                                  classname

        ************

        params:
            r_obj -> An istance of the Raster class
            gdf -> A geodataframe of point geometries
            outdir -> full path to output directory

        return:
           An instance of the Trainingdata class
        """
        # Create a (geometry, classname, id) geodf of all classes geodf
        # shapefiles = gpd.GeoDataFrame(pd.concat(list_gdfs,ignore_index=True, sort=False))

        # Numpy array of shapely objects
        geoms = gdf.geometry.values

        # convert all labels_id to int
        gdf.id = gdf.id.astype(int)

        # extract the raster values within the polygon
        with rasterio.open(r_obj.path_to_raster) as src:
            X = np.array([]).reshape(0,src.count)# pixels for training
            y = np.array([], dtype=np.int64) # labels_id for training
            #pdb.set_trace()
            # build (id, classname) mapping
            label_names = np.unique(gdf.classname)
            map_cl_id = {str(gdf.loc[gdf['classname'] == label_name, 'id'].iloc[0]): label_name for label_name in label_names}
            logger.debug('Classname to Label_id mapping:  {}'.format(map_cl_id))

            logger.debug('Extracting values from the raster...')

            for index, geom in enumerate(geoms):
                # Transform to GeoJSON format
                # [{'type': 'Point', 'coordinates': (746418.3300011896, 3634564.6338985614)}]
                feature = [mapping(geom)]

                # the mask function returns an array of the raster pixels within this feature
                # out_image.shape == (band_count,1,1) for one Point Object
                out_image, out_transform = mask(src, feature, crop=True)

                # reshape the array to [pixel values, band_count]
                out_image_reshaped = out_image.reshape(-1, src.count)
                # Checks for out_image_reshaped == 0
                # If equal to zero (masked) it will not be included in the final X,y
                if np.count_nonzero(out_image_reshaped) == 0:
                #    classname = gdf['classname'].iloc[index]
                #    logger.debug('''Point of class {} at dataframe index {},
                #                  not included in the training data because cloud masked'''.format(classname, index))
                    continue
                y = np.append(y,[gdf['id'][index]] * out_image_reshaped.shape[0])
                X = np.vstack((X,out_image_reshaped))

        logger.debug('Features array shape: {} Labels array shape {} '.format(X.shape, y.shape))

        if write:
            #pdb.set_trace()
            logger.debug('Writing training data to disk')
            json_arr = to_json({'X':X, 'y':y, 'map_cl_id':map_cl_id}, encoder=NumpyEncoder)
            json_to_disk(json_arr, 'training_data', outdir)

        return Trainingdata(X, y, map_cl_id)
