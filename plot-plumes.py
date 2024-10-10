#
# Author: Karlee Zammit
# Contact: karlee.zammit@nrcan-rncan.gc.ca
# Date: 2024-10-09

###################################################################################################
# Description
###################################################################################################

"""
Script to plot examples of known plumes. Plots:
- VIIRS hotspots (currently only using suomi-npp but will be updated to NOAA-21 and NOAA-20)
- Corresponding satellite imagery through GEE export/import:
    - Landsat 8 (toa reflectance, LANDSAT/LC08/C02/T1_TOA)
    - VIIRS VNP09GA (surface reflectance daily, NASA_VIIRS_002_VNP09GA)
- MTBS fire polygons for corresponding fires
    - Will update this to auto find the closest fire to the plume hotspots but currently was completed manually
"""

###################################################################################################
# Import statements
###################################################################################################

import ee
import geemap
import io
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
from cartopy import crs as ccrs
from geodatasets import get_path
import rioxarray as rxr
import matplotlib.patches as mpatches
import time
import seaborn as sns
import xarray

from glob import glob
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image

from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import google.auth
from googleapiclient.errors import HttpError

import os

from matplotlib.colors import ListedColormap

import plotly.graph_objects as go
from rasterio.plot import show

from PIL import Image
import numpy as np

from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

ee.Authenticate()

###################################################################################################
# Variables
###################################################################################################

### FIRMS Related Variables ###

# Map key
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'

# Coords of hotspots to pull, will also be used as bounding box for satellite imagery in GEE
coords = '-123,39,-121,41'

# Number of days surrounding date to pull hotspots
VIIRS_no_days = '1'

# Date to pull hotspots
VIIRS_date = '2018-07-26'

### GEE Related Variables ###

gee_proj = 'karlzam'

# DO NOT CHANGE THIS - hardcoded later on
# For other users: you need to share this folder with your credential for google API for google cloud
google_drive_folder = 'GEE_Exports'

start_date = '2018-07-25'

end_date = '2018-07-27'

download_path = r'C:\Users\kzammit\Documents\downloaded-files'

creds_file = r'C:\Users\kzammit\Documents\karlzam-d3258a83d6cb.json'

###################################################################################################
# Methods
###################################################################################################

def obtain_viirs_hotspots(FIRMS_map_key, coords, VIIRS_no_days, VIIRS_date):
    """

    Returns VIIRS hotspots from FIRMS for specified date, coordinates, and date range in the form of a
    geodataframe.

    :param FIRMS_map_key:
    :param VIIRS_coords:
    :param VIIRS_no_days:
    :param VIIRS_date:
    :return:
    """

    MAP_KEY = FIRMS_map_key
    url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
    try:
        df = pd.read_json(url, typ='series')
    except:
        # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
        print("There is an issue with the query. \nTry in your browser: %s" % url)

    # VIIRS_S-NPP
    area_url_SNPP = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_SNPP_SP/' +
                     str(coords) + '/' + str(VIIRS_no_days) + '/' + str(VIIRS_date))
    df_SNPP = pd.read_csv(area_url_SNPP)

    gdf_SNPP = gpd.GeoDataFrame(
        df_SNPP, geometry=gpd.points_from_xy(df_SNPP.longitude, df_SNPP.latitude), crs="EPSG:4326"
    )

    return gdf_SNPP


def download_gee_data(coords, start_date, end_date, google_drive_folder):
    """

    :param coords:
    :param start_date:
    :param end_date:
    :param google_drive_folder:
    :return:
    """

    roi = ee.Geometry.Rectangle([int(coords.split(',')[0]), int(coords.split(',')[1]),
                                 int(coords.split(',')[2]), int(coords.split(',')[3])])

    ## viirs
    # moderate resolution bands: 'M5', 'M4', 'M3'
    # moderate bands min max: 0, 0.3
    # imagery resolution bands: 'I1', 'I2', 'I3'
    # imagery resolution min max (note, KZ trial and error, no documentation): 0, 0.5

    viirs = ee.ImageCollection("NASA/VIIRS/002/VNP09GA").filterDate(start_date, end_date).filterBounds(roi)

    rgb_viirs = viirs.select(['I1', 'I2', 'I3'])

    #rgbVis_viirs = {'bands': ['I1', 'I2', 'I3'], 'min':0, 'max':0.5}

    image_viirs = rgb_viirs.median()

    clipped_viirs = image_viirs.clip(roi)

    ## export to google drive
    # This selects all bands that start with an "M" like in the example (but they had B)
    exportImage_viirs = clipped_viirs.select('I.*')

    export_gee_data(exportImage_viirs, roi, 'v', google_drive_folder)

    ## landsat

    landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").filterDate(start_date, end_date).filterBounds(roi)

    true_color_432 = landsat.select(['B4', 'B3', 'B2'])
    #true_color_432_vis = {'min': 0.0, 'max': 0.4}

    image_landsat = true_color_432.median()

    clipped_landsat = image_landsat.clip(roi)

    exportImage_landsat = clipped_landsat.select('B.*')

    export_gee_data(exportImage_landsat, roi, 'l', google_drive_folder)


def export_gee_data(exportImage, roi, flag, google_drive_folder):
    """

    :param exportImage:
    :param roi:
    :param flag:
    :param google_drive_folder:
    :return:
    """

    if flag=='v':
        # Define the export parameters
        export_task = ee.batch.Export.image.toDrive(
            image=exportImage,
            description='VIIRS_RGB_Export',
            folder=google_drive_folder,  # Change this to your preferred folder in Google Drive
            fileNamePrefix='viirs_rgb',
            region=roi,  # Define the region to export
            scale=500,  # Scale in meters
            crs='EPSG:4326',  # Coordinate reference system
            maxPixels=1e13  # Maximum number of pixels to export
        )

    elif flag=='l':
        # Define the export parameters
        export_task = ee.batch.Export.image.toDrive(
            image=exportImage,
            description='Landsat',
            folder=google_drive_folder,  # Change this to your preferred folder in Google Drive
            fileNamePrefix='landsat-truecolour',
            region=roi,  # Define the region to export
            scale=30,  # Scale in meters
            crs='EPSG:4326',  # Coordinate reference system
            maxPixels=1e13  # Maximum number of pixels to export
        )

    # Start the export task
    export_task.start()

    while export_task.active():
        print('Polling for task (id: {}).'.format(export_task.id))
        time.sleep(10)

    print('Task completed with status: ', export_task.status())


def download_gee_data():
    """
    This directly downloads files from my Google Drive in the GEE_Exports folder
    :return:
    """

    creds = service_account.Credentials.from_service_account_file(
        creds_file,
        scopes=['https://www.googleapis.com/auth/drive']
    )

    drive_service = build('drive', 'v3', credentials=creds)

    # First, get the folder ID by querying by mimeType and name
    folderId = drive_service.files().list(q="mimeType = 'application/vnd.google-apps.folder' and name = 'GEE_Exports'",
                                  pageSize=10, fields="nextPageToken, files(id, name)").execute()
    # this gives us a list of all folders with that name
    folderIdResult = folderId.get('files', [])
    # however, we know there is only 1 folder with that name, so we just get the id of the 1st item in the list
    id = folderIdResult[0].get('id')

    # Now, using the folder ID gotten above, we get all the files from
    # that particular folder
    results = drive_service.files().list(q="'" + id + "' in parents", pageSize=10,
                                 fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    for item in items:

        # NOTE: Had to give permission to certificate email to see the folder: google-drive@karlzam.iam.gserviceaccount.com
        # Replace this with getting all the files within the GEE folder
        path = download_path
        file_path = path + '\\' + str(item['name'])

        request = drive_service.files().get_media(fileId=item['id'])

        fh = io.FileIO(file_path, mode='wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False

        while not done:
            status, done = downloader.next_chunk()

def import_gee_data(download_path):
    """

    :param download_path:
    :return:
    """

    files = glob(download_path + '\\' + '*.tif')

    return files


if __name__ == "__main__":

    #ee.Initialize(project=gee_proj)
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    #viirs_hotspots = obtain_viirs_hotspots(FIRMS_map_key, coords, VIIRS_no_days, VIIRS_date)

    # WARNING: The Landsat scale is currently set to export at 50m, and exporting the .tif takes quite a while!
    #download_gee_data(coords, start_date, end_date, google_drive_folder)

    download_gee_data()

    files = import_gee_data(download_path)











