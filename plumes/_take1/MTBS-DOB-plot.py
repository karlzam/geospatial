#
# Author: Karlee Zammit
# Contact: karlee.zammit@nrcan-rncan.gc.ca
# Date: 2024-10-09

###################################################################################################
# Description
###################################################################################################


###################################################################################################
# Import statements
###################################################################################################

import ee
import io
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from shapely.geometry import MultiPoint, Polygon
import matplotlib.patches as mpatches
import numpy as np
from osgeo import gdal
from datetime import datetime, timedelta
import utm
from pyproj import CRS
import seaborn as sns
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.plot import show
import rioxarray

gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

###################################################################################################
# Methods
###################################################################################################

def locate_fire(row, MTBS):

    # LOAD IN PLUME DATA
    # Convert plume into a boundary polygon for plotting
    plume = gpd.read_file(row['shapefile'])

    # Create polygon for plume points
    poly = create_polygon(plume)

    # Get MTBS points from where the polygon intersects the fire boundary
    MTBS_filt = MTBS[MTBS['geometry'].intersects(poly)]

    # Create a new column to select year from the MTBS dataframe
    years = []
    for ii in range(0, len(MTBS_filt)):
        years.append(MTBS_filt.iloc[ii]['Ig_Date'].year)
    MTBS_filt['year'] = years

    # Pull out the specific fire for the year corresponding to the plume
    # (a fire could have happened in the same place in multiple years)
    fire = MTBS_filt.loc[MTBS_filt['year'] == row['plume-date'].year]
    fire = fire.reset_index()

    return plume, poly, fire


def create_polygon(shp):
    """

    :param shp:
    :return:
    """
    pts = shp['geometry']
    mp = MultiPoint(pts)
    conv_hull = mp.convex_hull
    poly = Polygon(conv_hull)

    return poly


if __name__ == "__main__":

    # downloaded from https://www.mtbs.gov/direct-download
    MTBS = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\MTBS\mtbs_perims_DD.shp')

    plume_df = pd.read_excel(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\plumes-for-mtbs.xlsx')

    for index, row in plume_df.iterrows():


        # Plume: VIIRS SNPP hotspots
        # Poly: Convex hull polygon around plume
        # Fire: MTBS fire with spatial overlap to the plume
        plume, poly, fire = locate_fire(row, MTBS)

        # Load the intermediate fire perimeters given by Piyush by event ID
        # One fire (Mineral) does not have corresponding ID
        if fire['Event_ID'][0] != 'CA3609512052220200713':

            #dob_tiff = rio.open(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\from-piyush' + '\\' +
            #                    str(fire['Event_ID'][0]) + '\\' + 'dob.tif')

            dob_tiff = rioxarray.open_rasterio(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\from-piyush' + '\\' +
                                str(fire['Event_ID'][0]) + '\\' + 'dob.tif', masked=True)
        else:

            #dob_tiff = rio.open(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\from-piyush' + '\\' +
            #                    'F566_2020' + '\\' + 'dob.tif')

            dob_tiff = rioxarray.open_rasterio(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\from-piyush' + '\\' +
                                'F566_2020' + '\\' + 'dob.tif', masked=True)


        dob_tiff = dob_tiff.rio.reproject("EPSG:4326")

        # Let's extract the day of burning parameters for these days
        plume_doy = row['plume-date'].to_pydatetime().timetuple().tm_yday
        dob_perim = np.full(dob_tiff.data.shape, np.nan)
        dob_perim[dob_tiff.data == plume_doy] = plume_doy

        dob_perim_tiff = dob_tiff.copy()
        dob_perim_tiff.data = dob_perim

        # Create a MTBS fire buffer to account for the fact that the hotspots might not actually
        # be in the center of the fire pixel
        # warning is ok because im just using this for the lat long
        centroid = fire['geometry'].centroid
        utm_fire = utm.from_latlon(centroid.y[0], centroid.x[0])
        crs = CRS.from_dict({'proj': 'utm', 'zone': utm_fire[2], 'south': False})

        fire_proj = fire.to_crs(crs.to_authority()[1])
        buffer_distance = 750
        fire_buff = fire_proj.copy()
        fire_buff['geometry'] = fire_proj.buffer(buffer_distance)
        fire_buff = fire_buff.to_crs(epsg=4326)

        # Get dataframe of spots outside of this buffered perimeter
        #df_outside_buffer = plume.iloc[:0, :].copy()
        #for index2, row2 in plume.iterrows():
        #    if not fire_buff.contains(row2['geometry'])[0]:
        #        df_outside_buffer.loc[len(df_outside_buffer)] = row2

        #df_inside_buffer = plume.iloc[:0, :].copy()
        #for index2, row2 in plume.iterrows():
        #    if fire_buff.contains(row2['geometry'])[0]:
        #        df_inside_buffer.loc[len(df_inside_buffer)] = row2


        # plot dob tif, plume, poly plume, and buffered MTBS boundary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
        #show(dob_tiff, ax=ax1)
        #fire['geometry'].plot(ax=ax1, color='black')
        dob_tiff.plot(cmap='viridis', ax=ax2)
        dob_perim_tiff.plot(cmap='inferno', ax=ax1)
        #fire_buff['geometry'].plot(ax=ax1, color='yellow')

        sns.scatterplot(x="longitude", y="latitude", data=plume, hue="frp", ax=ax1, size='frp',
                        palette='inferno_r')

        ax1.set_title('All Intermediate Boundaries')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')

        # int_per_plume.plot()
        path = r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries'
        plt.savefig(path + '\\' + str(fire['Incid_Name'][0]) + '-new.png', bbox_inches='tight')



















