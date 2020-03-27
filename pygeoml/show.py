import matplotlib.pyplot as plt
import numpy as np
import rasterio

def plot_select_class_prediction(class_prediction, color_map, classes, figsize=(10,10), fontsize=8, r_obj=None):

    # find the highest pixel value in the prediction image
    n = int(np.max(class_prediction[:,:,0]))


    # create a default white color map using colors as float 0-1
    index_colors = [(1.0, 1.0, 1.0) for key in range(0, n )]

    # Replace index_color with the one you want to visualize
    for cl in classes:
        idx = list(color_map[cl].keys())[0]
        vals = list(color_map[cl].values())[0]
        # Transform 0 - 255 color values from colors as float 0 - 1
        _v = [_v / 255.0 for _v in vals]
        index_colors[idx] = tuple(_v)

    cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', n)


    from matplotlib.patches import Patch
    # Create a list of labels sorted by the int of color_map
    class_labels = [el[0] for el in sorted(color_map.items(), key=lambda label: label[1].keys())]
    # A path is an object drawn by matplotlib. In this case a patch is a box draw on your legend
    # Below you create a unique path or box with a unique color - one for each of the labels above
    legend_patches = [Patch(color=icolor, label=label)
                      for icolor, label in zip(index_colors, class_labels)]

    # Plot Classification
    fig, axs = plt.subplots(1,1,figsize=figsize)
    axs.imshow(class_prediction[:,:,0], cmap=cmap, interpolation='none')
    if r_obj:
        from rasterio.plot import plotting_extent
        axs.imshow(class_prediction[:,:,0], extent=plotting_extent(r_obj), cmap=cmap, interpolation='none')

    axs.legend(handles=legend_patches,
              facecolor="white",
              edgecolor="white",
              bbox_to_anchor=(1.20, 1),
              fontsize=fontsize)  # Place legend to the RIGHT of the map
    axs.set_axis_off()
    plt.show()

    def show_hist(path_to_raster):
        """
        (to be modified)
        """
        with rasterio.open(path_to_raster) as dataset:
            rasterio.plot.show_hist(dataset.read([1,2,3,4]), bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)

#    def show(self, **kwargs):
#        """
#        Show a first raster layer
#
#        *********
#
#        keyword args:
#            height -- raster height (default full height)
#            width -- raster width (default full width)
#            bands -- band to show, for multiple bands e.g. [1,2,3] (default 0)
#            coll_off -- starting column (default 0)
#            row_off -- starting row (default 0)
#            fcmap -- colormap (defaulf "pink")
#            fsize -- figure size (default 10)
#            fbar -- figure colorbar (default False)
#            fclim -- figure colorbar range (default None)
#
#        """
#
#        height, width, bands, col_off, row_off, fcmap, fsize, fbar, fclim = kwargs.get('height', self.height),\
#                             kwargs.get('width', self.width),\
#                             kwargs.get('bands', 0), kwargs.get('col_off', 0),\
#                             kwargs.get('row_off', 0), kwargs.get('fcmap','pink'),\
#                             kwargs.get('fsize', 10), kwargs.get('fbar', False),\
#                             kwargs.get('fclim', None)
#
#        arr = self.load_as_arr()
#
#        # Plotting
#        fig, ax = plt.subplots(figsize=(fsize, fsize))
#
#        if isinstance(bands, int):
#            if not fclim:
#                fclim = (np.min(arr), np.max(arr))
#            img = ax.imshow(arr[:,:,bands], cmap=fcmap)
#            img.set_clim(vmin=fclim[0],vmax=fclim[1])
#        else:
#            img = ax.imshow(arr, cmap=fcmap)
#        if fbar:
#            fig.colorbar(img, ax=ax)
#
#    @classmethod
#    def points_on_layer_plot(self, r_obj, layer_arr, gdf, band=0, **kwargs):
#
#        layer_endrow = layer_arr.shape[0]
#        layer_endcol = layer_arr.shape[1]
#        layer_poly = [r_obj.transform_to_coordinates(0,0), r_obj.transform_to_coordinates(layer_endrow,0),
#                      r_obj.transform_to_coordinates(layer_endrow,layer_endcol), r_obj.transform_to_coordinates(0,layer_endcol)]
#
#        # check for raster and arr coordinates
#        assert r_obj.get_raster_polygon() == Polygon(layer_poly), "Input array and raster must have same coordinates"
#
#        cmap, marker, markersize, color, label = kwargs.get('r_cmap',"pink"), \
#                                  kwargs.get('s_marker',"s"), \
#                                  kwargs.get('s_markersize',30), \
#                                  kwargs.get('s_color',"purple"), \
#                                  kwargs.get('s_label',"classname")
#
#        # Plotting
#        fig, ax = plt.subplots(figsize=(10, 10))
#
#        ax.imshow(layer_arr[:,:,band],
#                      # Set the spatial extent or else the data will not line up with your geopandas layer
#                      extent=plotting_extent(r_obj),
#                      cmap=cmap)
#        gdf.plot(ax=ax,
#                     marker=marker,
#                     markersize=markersize,
#                     color=color,
#                     label=label)
#        return ax
