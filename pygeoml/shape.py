import os
import copy

import geopandas as gpd
import pandas as pd
import numpy as np

class Shapeobj():
    """
    Build ShapeObj from a shapefile.

    A ShapeObj is a Geopandas dataframe with a
    classname (shapefile name) and an id (instance number)
    columns added to it

    ***********************

    Attributes
    ---------

    path: full path to the shapefile

    gpd: geodataframe; accessed using instance.gpd

    classname: shapefile name without the .shp extension.
               Added as a column to the gpd.

    id: id of the instance. Added as a column to the gpd.


    """

    idCounter = 0 # bin counter for instances

    def __init__(self, path):
        self.path = path
        self.classname = os.path.basename(path).split('.')[0]
        self.gdf = gpd.read_file(path)
        # Increment each time I create a new instance
        Shapeobj.idCounter += 1
        self.id = Shapeobj.idCounter
        # Add id and classname columns
        self.gdf['classname'] = [self.classname for i in range(self.gdf.shape[0])]
        self.gdf['id'] = [self.id for i in range(self.gdf.shape[0])]

    def __repr__(self):
       return "Shapeobj({})".format(self.path)

    def drop_columns(self, *argv):
        # This will change gdf in place
        self.gdf.drop([*argv],1,inplace=True)

    def reproject(self, ref):
        # This will change gdf in place
        self.gdf = self.gdf.to_crs({'init':ref})


