import os

import rasterio
import rasterio.plot
from rasterio.mask import mask
from rasterio.merge import merge
import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import mapping
from pygeoml.utils import np_to_disk

class Trainingdata:
    """
    Generate/modify:
      X features
      y labels
    used as training data for various ML algorithms.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def save_xy(self, rpath, outdir):
        np_to_disk(self.X, 'features', rpath, outdir)
        np_to_disk(self.y, 'lables', rpath, outdir)

    def add_class_xy(self, class_name, class_value, n=80):
        """
        Extend X,y in the 0 dimension by adding a n number of
        class_value to X and a n numbers of class_name label to y.

        n is a number of objects to stack to X and y
        """
        class_val = np.ones((n, self.X.shape[1]))*class_value
        class_label = np.array([class_name for _ in range(n)])
        self.X = np.vstack((self.X,class_val))
        self.y = np.append(self.y,class_label)

    def hstack_to_x(self, other):
        # check x for equal 0 dimension
        assert self.X.shape[0] == other.X.shape[0]
        # check y for equal labels
        np.testing.assert_array_equal(self.y,other.y)
        # create a final array
        final_x = np.hstack((self.X,other.X))
        return Trainingdata(final_x,self.y)

    def build_str_to_int_map(self):
        classnames = list(set(self.y))
        self.str_int_map = {classnames[idx]:idx for idx in range(0,len(classnames))}

    @classmethod
    def to_df(cls, X, y):
        # check X feature and y labels have the same 0 dimension
        assert X.shape[0] == y.shape[0], "X and y should have the same 0 dimension"
        df = pd.DataFrame(data=X,
                          index=[i for i in range(0, X.shape[0])],
                          columns=["Band"+str(i) for i in range(1, X.shape[1]+1)])
        df['labels'] = y.tolist()
        return df

    @classmethod
    def load_xy(cls, path_to_x, path_to_y):
        X = np.load(path_to_x)
        y = np.load(path_to_y)
        return Trainingdata(X,y)

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

        # extract the raster values within the polygon
        with rasterio.open(r_obj.path_to_raster) as src:
            X = np.array([]).reshape(0,src.count)# pixels for training
            y = np.array([], dtype=np.string_) # labels for training
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
                    continue
                y = np.append(y,[gdf["classname"][index]] * out_image_reshaped.shape[0])
                X = np.vstack((X,out_image_reshaped))

        if write:
            np_to_disk(X, 'features', r_obj.path_to_raster, outdir=outdir)
            np_to_disk(y, 'lables', r_obj.path_to_raster, outdir=outdir)

        return Trainingdata(X,y)

    @staticmethod
    def predict(r_obj, v_split, h_split, classifier, write=False, outdir=None):
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
        class_final = np.empty((r_obj.height,r_obj.width), dtype="S10")

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
            np_to_disk(class_final, 'class_prediction', r_obj.path_to_raster, outdir=outdir)
        return class_final
