import os
import geopandas as gpd

# Enable the read and write functionalities for KML-driver by passing
# 'rw' to whitelist of fiona’s supported drivers:
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'




class Shapeobj():
    """

    Build ShapeObj from a shapefile.

    A ShapeObj is a Geopandas dataframe with a
    classname (shapefile name) and an id (instance number)
    columns added to it

    Note: It assumes each .shp file has a SINGLE class

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
        self.gdf = gpd.read_file(path)
        # Increment each time I create a new instance
        Shapeobj.idCounter += 1
        self.id = Shapeobj.idCounter

    def __repr__(self):
       return "Shapeobj({})".format(self.path)

    def drop_columns(self, *argv):
        # This will change gdf in place
        self.gdf.drop([*argv],1,inplace=True)

    def reproject(self, ref):
        # This will change gdf in place
        self.gdf = self.gdf.to_crs({'init':ref})

    def write_gdf(self):
        pass

    def make_class_and_id(self):
        """
        Insert class name and ID columns into a data frame
        """
        self.classname = os.path.basename(self.path).split('.')[0]
        self.gdf['classname'] = [self.classname for i in range(self.gdf.shape[0])]
        # Add id and classname columns
        self.gdf['id'] = [self.id for i in range(self.gdf.shape[0])]

# assign value to a column based of values of other column

#set(new_gdf['classname']) =
#{'Agriculture',
#  'Agricutlure',
#  'Bare rock',
#  'Building',
#  'Farm',
#  'Forest',
#  'Livestock',
#  'Marsh',
#  'River',
#  'Road',
#  'Tree plantation'}
#
#ids=
#[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#
#for classname,i in zip(set(new_gdf['classname']),ids):
#        new_gdf.loc[new_gdf['classname'] == classname, 'id'] = i
#
#non_charcol_obj.crs
#
#type(non_charcol_obj)
#
#new_gdf = non_charcol_obj.rename(columns = {'Name':'classname'})
#
#new_gdf[new_gdf['classname']=='Builidng'] = 'Building'
#
#new_gdf[new_gdf['classname']=='Builiding'] = 'Building'
#
#set(new_gdf['classname'])
