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


def plot_MTBS_plumes():

    # Load the intermediate fire perimeters given by Piyush by event ID
    # One fire (Mineral) does not have corresponding ID
    if fire['Event_ID'][0] != 'CA3609512052220200713':
        int_per = gpd.read_file(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\from-piyush' + '\\' +
                                str(fire['Event_ID'][0]) + '\\' + str(fire['Event_ID'][0]) + '_hotspots.shp')
    else:
        int_per = gpd.read_file(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\from-piyush' + '\\' +
                                'F566_2020' + '\\' + 'F566_2020' + '_hotspots.shp')

    # filter intermediate parameter points to be only plume date
    int_per_plume = int_per[int_per['ACQ_DATE']==row['plume-date']]
    int_per_plume = int_per_plume.reset_index()

    # create polygon from intermediate perimeter corresponding to plume date
    poly_int = create_polygon(int_per_plume)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)

    # Plotting
    fire['geometry'].plot(ax=ax1, color='black')
    fire['geometry'].plot(ax=ax2, color='black')
    fire['geometry'].plot(ax=ax3, color='black')

    unq_dates = int_per['ACQ_DATE'].unique()
    color = cm.plasma(np.linspace(0, 1, len(unq_dates)))

    for ii in range(0, len(int_per['ACQ_DATE'].unique())):
        temp = int_per[int_per['ACQ_DATE']==unq_dates[ii]]
        int_per_temp = temp.reset_index()
        # can't make a polygon for less than 4 coords
        if len(int_per_temp)>=4:
            poly_int_temp = create_polygon(int_per_temp)
            ax1.plot(*poly_int_temp.exterior.xy, label='other-dates', color=color[ii])

    dm1 = row['plume-date'] - timedelta(days=1)
    dp1 = row['plume-date'] + timedelta(days=1)

    int_per_m1 = int_per[int_per['ACQ_DATE']==dm1]
    int_per_m1 = int_per_m1.reset_index()
    if len(int_per_m1)>=4:
        poly_int_m1 = create_polygon(int_per_m1)

    int_per_p1 = int_per[int_per['ACQ_DATE'] == dp1]
    int_per_p1 = int_per_p1.reset_index()
    if len(int_per_p1)>=4:
        poly_int_p1 = create_polygon(int_per_p1)

    if len(int_per_m1)>=4:
        ax3.plot(*poly_int_m1.exterior.xy, label='plume', color='#404788FF')
    if len(int_per_p1)>=4:
        ax3.plot(*poly_int_p1.exterior.xy, label='plume', color='#55C667FF')
    ax3.plot(*poly.exterior.xy, label='plume', color='red')
    ax3.set_title('+- Day Perimeters')

    ax2.scatter(plume.longitude, plume.latitude, color='red', s=1)

    ax1.set_title('All Intermediate Boundaries')
    ax2.plot(*poly.exterior.xy, label='plume', color='red')
    ax2.plot(*poly_int.exterior.xy, label='int_per', color='yellow')
    ax2.set_title(str(fire['Incid_Name'][0]) + ' Plume')
    ax2.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    red_patch = mpatches.Patch(color='red', label='Plume-Per')
    yellow_patch = mpatches.Patch(color='yellow', label='Int-Per-Day-Of')
    black_patch = mpatches.Patch(color='black', label='MTBS-Per')
    p1_patch = mpatches.Patch(color='#55C667FF', label='Int-Per-+1-Day')
    m1_patch = mpatches.Patch(color='#404788FF', label='Int-Per--1-Day')
    plt.legend(handles=[black_patch, red_patch, yellow_patch, p1_patch, m1_patch], bbox_to_anchor=(1.05, 1), loc='upper left')

    path = r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries'
    plt.savefig(path + '\\' + str(fire['Incid_Name'][0]) + '.png', bbox_inches='tight')


if __name__ == "__main__":

    # downloaded from https://www.mtbs.gov/direct-download
    MTBS = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\MTBS\mtbs_perims_DD.shp')

    plume_df = pd.read_excel(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\plumes-for-mtbs.xlsx')

    for index, row in plume_df.iterrows():

        plume, poly, fire = locate_fire(row, MTBS)

        # plot_MTBS_plumes()

        # Apply 750 m buffer to final fire perimeter

        # warning is ok because im just using this for the lat long
        centroid = fire['geometry'].centroid
        utm_fire = utm.from_latlon(centroid.y[0], centroid.x[0])
        crs = CRS.from_dict({'proj': 'utm', 'zone': utm_fire[2], 'south': False})

        fire_proj = fire.to_crs(crs.to_authority()[1])
        buffer_distance = 750
        fire_buff = fire_proj.copy()
        fire_buff['geometry'] = fire_proj.buffer(buffer_distance)
        fire_buff = fire_buff.to_crs(epsg=4326)

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
        fire_buff['geometry'].plot(ax=ax1, color='yellow')
        fire['geometry'].plot(ax=ax1, color='black')
        ax1.scatter(plume.longitude, plume.latitude, color='red', s=1)

        # Get dataframe of spots outside of this buffered perimeter
        df_outside_buffer = plume.iloc[:0, :].copy()
        for index2, row2 in plume.iterrows():
            if not fire_buff.contains(row2['geometry'])[0]:
                df_outside_buffer.loc[len(df_outside_buffer)] = row2

        df_inside_buffer = plume.iloc[:0, :].copy()
        for index2, row2 in plume.iterrows():
            if fire_buff.contains(row2['geometry'])[0]:
                df_inside_buffer.loc[len(df_inside_buffer)] = row2

        print('test')


        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
        fire_buff['geometry'].plot(ax=ax1, color='yellow')
        fire['geometry'].plot(ax=ax1, color='black')
        #ax1.scatter(plume.longitude, plume.latitude, color='red', s=1)
        #if len(df_outside_buffer) > 0:
        #    print('test')
        #    df_outside_buffer['geometry'].plot(ax=ax1, color='grey')
        #    sns.scatterplot(x="longitude", y="latitude", data=df_outside_buffer, hue="frp", ax=ax1, size='frp', palette='viridis')
        #sns.scatterplot(x="longitude", y="latitude", data=df_inside_buffer, hue="frp", ax=ax1, size='frp',
        #                palette='plasma')

        sns.scatterplot(x="longitude", y="latitude", data=plume, hue="frp", ax=ax1, size='frp',
                        palette='plasma')

        #int_per_plume.plot()
        path = r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries'
        plt.savefig(path + '\\' + str(fire['Incid_Name'][0]) + '-buffer.png', bbox_inches='tight')


        print('test')


        # Look at FRP of the spots outside vs inside the perimeter
















