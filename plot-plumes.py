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
import openpyxl

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

plume_excel_sheet = pd.read_excel(r'C:\Users\kzammit\Documents\Plumes\plume-metadata.xlsx')

plot_path = r'C:\Users\kzammit\Documents\Plumes\plots'

### FIRMS Related Variables ###

# Map key
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'

### GEE Related Variables ###

gee_proj = 'karlzam'

# DO NOT CHANGE THIS - hardcoded later on
# For other users: you need to share this folder with your credential for google API for google cloud
google_drive_folder = 'GEE_Exports'

# Where satellite data will be exported to
download_path = r'C:\Users\kzammit\Documents\Plumes\downloaded-tifs'

# Before running this script:
# Create a service key for your google cloud project and download the .json to your local machine
creds_file = r'C:\Users\kzammit\Documents\google-drive-API-key\karlzam-d3258a83d6cb.json'


###################################################################################################
# Methods
###################################################################################################

def get_info(r):
    # Filled in by sheet

    id = str(r['id'])

    # Coords of hotspots to pull, will also be used as bounding box for satellite imagery in GEE
    coords = str(r['coord-0']) + ',' + str(r['coord-1']) + ',' + str(r['coord-2']) + ',' + str(r['coord-3'])

    # Number of days surrounding date to pull hotspots
    VIIRS_no_days = int(r['viirs-no-days'])

    # Date to pull hotspots
    VIIRS_date = str(r['viirs-date']).split(' ')[0]

    # Start and end dates for filtering in google
    start_date = str(r['gee-start-date']).split(' ')[0]
    end_date = str(r['gee-end-date']).split(' ')[0]

    return id, coords, VIIRS_no_days, VIIRS_date, start_date, end_date

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


def download_gee_data(id, coords, start_date, end_date, google_drive_folder, landsat_flag):
    """

    :param coords:
    :param start_date:
    :param end_date:
    :param google_drive_folder:
    :return:
    """

    # TODO: Update the image collection to grab the closest date image to the plume date instead of taking the median

    roi = ee.Geometry.Rectangle([float(coords.split(',')[0]), float(coords.split(',')[1]),
                                 float(coords.split(',')[2]), float(coords.split(',')[3])])

    ## viirs
    # moderate resolution bands: 'M5', 'M4', 'M3'
    # moderate bands min max: 0, 0.3
    # imagery resolution bands: 'I1', 'I2', 'I3'
    # imagery resolution min max (note, KZ trial and error, no documentation): 0, 0.5

    viirs = ee.ImageCollection("NASA/VIIRS/002/VNP09GA").filterDate(start_date, end_date).filterBounds(roi)

    #rgb_viirs = viirs.select(['I1', 'I2', 'I3'])
    rgb_viirs = viirs.select(['M5', 'M4', 'M3'])

    #rgbVis_viirs = {'bands': ['I1', 'I2', 'I3'], 'min':0, 'max':0.5}

    image_viirs = rgb_viirs.median()

    clipped_viirs = image_viirs.clip(roi)

    ## export to google drive
    # This selects all bands that start with an "M" like in the example (but they had B)
    exportImage_viirs = clipped_viirs.select('M.*')

    export_gee_data(id, exportImage_viirs, roi, 'v', google_drive_folder)

    ## landsat

    if landsat_flag == 1:

        landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").filterDate(start_date, end_date).filterBounds(roi)

        true_color_432 = landsat.select(['B4', 'B3', 'B2'])
        #true_color_432_vis = {'min': 0.0, 'max': 0.4}

        image_landsat = true_color_432.median()

        clipped_landsat = image_landsat.clip(roi)

        exportImage_landsat = clipped_landsat.select('B.*')

        export_gee_data(id, exportImage_landsat, roi, 'l', google_drive_folder)


def export_gee_data(id, exportImage, roi, flag, google_drive_folder):
    """
    Export data from google earth engine to google drive
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
            fileNamePrefix= id + '-viirs_rgb',
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
            fileNamePrefix= id + '-landsat-truecolour',
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


def drive_download_data():
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

def plot_plume(id, hotspots, tif_folder, plot_folder):
    """

    :param hotspots:
    :param tif_folder:
    :param plot_folder:
    :return:
    """

    files = glob(tif_folder + '\\' + '*.tif')

    sub_files = []
    for ii in range(0, len(files)):
        if id in files[ii]:
            sub_files.append(files[ii])

    for tif_file in sub_files:

        if 'viirs' in tif_file:

            source = 'viirs'
            plot_rgb(id, tif_file, hotspots, plot_folder, source)

        if 'landsat' in tif_file:

            source = 'landsat'
            plot_rgb(id, tif_file, hotspots, plot_folder, source)



def plot_rgb(id, tif_file, hotspots, plot_folder, source):

    with rio.open(tif_file) as src:
        # Read the image data
        # img_data = src.read(1)
        b1 = src.read(1)
        b2 = src.read(2)
        b3 = src.read(3)

        rgb = np.dstack((b1, b2, b3))

        # Get the metadata
        transform = src.transform
        bounds = src.bounds

        # Plotting
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        plt.scatter(hotspots['longitude'], hotspots['latitude'], label='VIIRS-SNPP Hot Spots')
        plt.title(tif_file.split('\\')[-1])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.savefig(plot_folder + '\\' + str (id) + '-' + str(source) +  '.png')


if __name__ == "__main__":

    #ee.Initialize(project=gee_proj)
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

    for ii in range(0, len(plume_excel_sheet)):

        print('reading excel sheet')
        id, coords, VIIRS_no_days, VIIRS_date, start_date, end_date = get_info(plume_excel_sheet.iloc[ii])

        print('obtaining VIIRS hotspots')
        viirs_hotspots = obtain_viirs_hotspots(FIRMS_map_key, coords, VIIRS_no_days, VIIRS_date)

        print('accessing and downloading gee imagery to drive')
        # WARNING: The Landsat scale is currently set to export at 50m, and exporting the .tif takes quite a while!
        landsat_flag = 1
        download_gee_data(id, coords, start_date, end_date, google_drive_folder, landsat_flag)

        print('downloading data from drive to local path')
        drive_download_data()

        plot_plume(id, viirs_hotspots, download_path, plot_path)
        print('Completed fire ' + str(id))













