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
from shapely.geometry import MultiPoint, Polygon
import matplotlib.patches as mpatches
from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')


###################################################################################################
# Variables
###################################################################################################

# Create a service key for your google cloud project and download the .json to your local machine
creds_file = r'C:\Users\kzammit\Documents\google-drive-API-key\karlzam-d3258a83d6cb.json'


###################################################################################################
# Methods
###################################################################################################


if __name__ == "__main__":
    #ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    # Started with google earth engine but I don't want to deal with the import/export
    #ds = ee.FeatureCollection('USFS/GTAC/MTBS/burned_area_boundaries/v1')
    # Converts to dataframe from feature collection
    #gdf = geemap.ee_to_df(ds)
    #carr = ds.filter(ee.Filter.eq('Event_ID', 'CA4065012263020180723'))

    # downloaded from https://www.mtbs.gov/direct-download
    MTBS = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\MTBS\mtbs_perims_DD.shp')

    plume_df = pd.read_excel(r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries\plumes-for-mtbs.xlsx')

    for index, row in plume_df.iterrows():

        # Convert plume into a boundary polygon for plotting
        plume = gpd.read_file(row['shapefile'])
        # https://stackoverflow.com/questions/60194404/how-to-make-a-polygon-shapefile-which-corresponds-to-the-outer-boundary-of-the-g
        pts = plume['geometry']
        mp = MultiPoint(pts)
        conv_hull = mp.convex_hull
        poly = Polygon(conv_hull)

        # Get MTBS points from where the polygon intersects the fire boundary
        MTBS_filt = MTBS[MTBS['geometry'].intersects(poly)]

        years = []
        for ii in range(0, len(MTBS_filt)):
            years.append(MTBS_filt.iloc[ii]['Ig_Date'].year)

        MTBS_filt['year'] = years

        fire = MTBS_filt.loc[MTBS_filt['year'] == row['plume-date'].year]
        fire=fire.reset_index()

        # Plotting
        plt.figure(figsize=(10, 10))
        fire['geometry'].plot(color='blue')
        plt.plot(*poly.exterior.xy, label='plume', color='red')
        plt.title('Plume Corresponding to ' + str(fire['Incid_Name'][0]) + ' Fire')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        blue_patch = mpatches.Patch(color='blue', label=str(fire['Incid_Name'][0]))
        red_patch = mpatches.Patch(color='red', label='Plume')
        plt.legend(handles=[blue_patch, red_patch], bbox_to_anchor=(1.05, 1), loc='upper left')

        path = r'C:\Users\kzammit\Documents\Plumes\MTBS-boundaries'
        plt.savefig(path + '\\' + str(fire['Incid_Name'][0]) + '.png', bbox_inches='tight')













