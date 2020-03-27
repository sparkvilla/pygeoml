import os
import geopandas as gpd

# Enable the read and write functionalities for KML-driver by passing
# 'rw' to whitelist of fiona’s supported drivers:
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'


class Shapeobj():
    """
    (to be changed)
    """
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

    def write_gdf(self, fname=None, outdir=None):
        if not fname:
            fname = os.path.basename(self.path).split('.')[0]+'_dataframe'
        if not outdir:
            # set outdir as the input raster location
            outdir = os.path.dirname(self.path)
        self.gdf.to_file(os.path.join(outdir,fname))

    def make_class_and_id(self):
        """
        Insert class name and ID columns into a data frame
        """
        self.classname = os.path.basename(self.path).split('.')[0]
        self.gdf['classname'] = [self.classname for i in range(self.gdf.shape[0])]
        # Add id and classname columns
        self.gdf['id'] = [self.id for i in range(self.gdf.shape[0])]

    def get_classes_df(self):
        ls = []
        classes = set(self.gdf['classname'])
        for cl in classes:
            df = self.gdf[self.gdf['classname']==cl]
            ls.append(df)
        return ls

    def gdf_within_polygon(self, poly):
        """
        Update gdf attribute with a new geodataframe with Points only within the
        Raster polygon
        """
        # create a mask
        mask = self.gdf.within(poly)
        gdf_within = self.gdf.loc[mask]
        self.gdf = gdf_within.reset_index(drop=True)

    def rename_field(self, name, new_name):
        """
        Update gdf attribute by replacing a classname field
        """
        self.gdf.replace(name, new_name, inplace=True)
