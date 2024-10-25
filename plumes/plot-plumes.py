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
import io
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import time
from timezonefinder import TimezoneFinder
from glob import glob
import rasterio as rio
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
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


def obtain_hotspots(id, FIRMS_map_key, coords, VIIRS_no_days, VIIRS_date):
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

    #gdf_SNPP.to_file(r'C:\Users\kzammit\Documents\Plumes\VIIRS-hotspots' + '\\' + str(id) + '-snpp.shp')

    area_url_MODIS = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/MODIS_SP/' +
                     str(coords) + '/' + str(VIIRS_no_days) + '/' + str(VIIRS_date))

    df_MODIS = pd.read_csv(area_url_MODIS)

    gdf_MODIS = gpd.GeoDataFrame(
        df_MODIS, geometry=gpd.points_from_xy(df_MODIS.longitude, df_MODIS.latitude), crs="EPSG:4326"
    )

    #gdf_MODIS.to_file(r'C:\Users\kzammit\Documents\Plumes\VIIRS-hotspots' + '\\' + str(id) + '-modis.shp')

    print('test')

    return gdf_SNPP, gdf_MODIS


def apply_scale_and_offset(image):
    # Band aliases.
    BLUE = 'CMI_C01'
    RED = 'CMI_C02'
    VEGGIE = 'CMI_C03'
    GREEN = 'GREEN'

    # Number of bands in the EE asset, 0-based.
    NUM_BANDS = 33

    # Skipping the interleaved DQF bands.
    BLUE_BAND_INDEX = (1 - 1) * 2
    RED_BAND_INDEX = (2 - 1) * 2
    VEGGIE_BAND_INDEX = (3 - 1) * 2
    GREEN_BAND_INDEX = NUM_BANDS - 1

    # Visualization range for GOES RGB.
    GOES_MIN = 0.0
    GOES_MAX = 0.7  # Alternatively 1.0 or 1.3.
    GAMMA = 1.3
    bands = [None] * NUM_BANDS  # Initialize with None to ensure correct length.

    for i in range(1, 17):
        band_name = f'CMI_C{str(100 + i)[-2:]}'
        offset = ee.Number(image.get(f'{band_name}_offset'))
        scale = ee.Number(image.get(f'{band_name}_scale'))
        bands[(i - 1) * 2] = image.select(band_name).multiply(scale).add(offset)

        dqf_name = f'DQF_C{str(100 + i)[-2:]}'
        bands[(i - 1) * 2 + 1] = image.select(dqf_name)

    # Green = 0.45 * Red + 0.10 * NIR + 0.45 * Blue
    green1 = bands[RED_BAND_INDEX].multiply(0.45)
    green2 = bands[VEGGIE_BAND_INDEX].multiply(0.10)
    green3 = bands[BLUE_BAND_INDEX].multiply(0.45)
    green = green1.add(green2).add(green3)
    bands[GREEN_BAND_INDEX] = green.rename(GREEN)

    return ee.Image(ee.Image(bands).copyProperties(image, image.propertyNames()))


def download_gee_data(id, coords, start_date, end_date, hotspot_date, landsat_flag):
    """

    :param id:
    :param coords:
    :param hotspot_date:
    :param google_drive_folder:
    :param landsat_flag:
    :return:
    """

    # TODO: Update the image collection to grab the closest date image to the plume date instead of taking the median

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

    date_of_interest = ee.Date(hotspot_date)

    ## VIIRS TRUE COLOUR
    viirs = ee.ImageCollection("NASA/VIIRS/002/VNP09GA").filterDate(start_date, end_date).filterBounds(roi)

    rgb_viirs_tc = viirs.select(['M5', 'M4', 'M3'])

    # Subtract the time of each image in collection from date of interest
    rgb_viirs_tc_sort = rgb_viirs_tc.map(lambda image: image.set(
        'dateDist',
        ee.Number(image.get('system:time_start')).subtract(date_of_interest.millis()).abs()
    ))

    # sort in ascending order by dateDist (so top image will correspond to date of interest)
    viirs_ic_rc_sorted = rgb_viirs_tc_sort.sort('dateDist')

    # grab the first image from the sorted image collection
    img_viirs_tc = viirs_ic_rc_sorted.first()

    # clip the image to the roi
    clipped_viirs_tc = img_viirs_tc.clip(roi)

    # export using visualization parameters suggested by GEE
    export_image_viirs_tc = clipped_viirs_tc.select('M.*').visualize(min=0, max=0.4)
    export_gee_data(id, export_image_viirs_tc, roi, 'v1')

    ## VIIRS FALSE COLOUR
    rgb_viirs_fc = viirs.select(['I3', 'I2', 'I1'])

    # Subtract the time of each image in collection from date of interest
    rgb_viirs_fc_sort = rgb_viirs_fc.map(lambda image: image.set(
        'dateDist',
        ee.Number(image.get('system:time_start')).subtract(date_of_interest.millis()).abs()
    ))

    # sort in ascending order by dateDist (so top image will correspond to date of interest)
    viirs_ic_fc_sorted = rgb_viirs_fc_sort.sort('dateDist')

    # grab the first image from the sorted image collection
    img_viirs_fc = viirs_ic_fc_sorted.first()

    # clip the image to the roi
    clipped_viirs_fc = img_viirs_fc.clip(roi)

    # export using visualization parameters suggested by GEE
    export_image_viirs_fc = clipped_viirs_fc.select('I.*').visualize(min=0, max=0.4)
    export_gee_data(id, export_image_viirs_fc, roi, 'v2')

    ## MODIS TERRA
    terra_c = ee.ImageCollection('MODIS/061/MOD09GA').filterDate(start_date, end_date).filterBounds(roi)
    terra = terra_c.select(['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03'])

    # Subtract the time of each image in collection from date of interest
    terra_sort = terra.map(lambda image: image.set(
        'dateDist',
        ee.Number(image.get('system:time_start')).subtract(date_of_interest.millis()).abs()
    ))

    # sort in ascending order by dateDist (so top image will correspond to date of interest)
    terra_sorted = terra_sort.sort('dateDist')

    # grab the first image from the sorted image collection
    img_terra = terra_sorted.first()

    # clip the image to the roi
    clipped_terra = img_terra.clip(roi)

    # export using visualization parameters suggested by GEE
    export_image_terra = clipped_terra.select('s.*').visualize(min=-100, max=8000)
    export_gee_data(id, export_image_terra, roi, 't')

    ## MODIS AQUA
    aqua_c = ee.ImageCollection('MODIS/061/MYD09GA').filterDate(start_date, end_date).filterBounds(roi)
    aqua = aqua_c.select(['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03'])

    # Subtract the time of each image in collection from date of interest
    aqua_sort = aqua.map(lambda image: image.set(
        'dateDist',
        ee.Number(image.get('system:time_start')).subtract(date_of_interest.millis()).abs()
    ))

    # sort in ascending order by dateDist (so top image will correspond to date of interest)
    aqua_sorted = aqua_sort.sort('dateDist')

    # grab the first image from the sorted image collection
    img_aqua = aqua_sorted.first()

    # clip the image to the roi
    clipped_aqua = img_aqua.clip(roi)

    # export using visualization parameters suggested by GEE
    export_image_aqua = clipped_aqua.select('s.*').visualize(min=-100, max=8000)
    export_gee_data(id, export_image_aqua, roi, 'a')


    # GOES

    goes = ee.ImageCollection("NOAA/GOES/16/MCMIPC").filterDate(start_date, end_date).filterBounds(roi)

    # Get timezone of coords (using first pair of coords)
    obj = TimezoneFinder()
    timezone = obj.timezone_at(lng=float(coords.split(',')[0]), lat=float(coords.split(',')[1]))

    # Want to do 1:30 pm local time to coincide with viirs overpass
    goes_date = ee.Date(hotspot_date + 'T13:30:00', timezone)

    goes_sort = goes.map(lambda image: image.set(
            'dateDist',
            ee.Number(image.get('system:time_start')).subtract(goes_date.millis()).abs()))

    # sort in ascending order by dateDist (so top image will correspond to date of interest)
    goes_sorted = goes_sort.sort('dateDist')

    # grab the first image from the sorted image collection
    goes_img = goes_sorted.first()

    # clip the image to the roi
    goes_img_clipped = goes_img.clip(roi)

    goes_img_tc = apply_scale_and_offset(ee.Image(goes_img_clipped))

    export_image_goes = goes_img_tc.select(['CMI_C02', 'GREEN', 'CMI_C01']).visualize(min=0, max=0.7, gamma=1.3)
    export_gee_data(id, export_image_goes, roi, 'g')

    ## landsat
    if landsat_flag == 1:

        landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").filterDate(start_date, end_date).filterBounds(roi)

        true_color_432 = landsat.select(['B4', 'B3', 'B2'])
        #true_color_432_vis = {'min': 0.0, 'max': 0.4}

        # Because landsat is very slow, we want to use as much data as possible
        # So we'll take the median value instead of sorting by date like we did for daily viirs
        image_landsat = true_color_432.median()

        clipped_landsat = image_landsat.clip(roi)

        export_image_landsat = clipped_landsat.select('B.*').visualize(min=0, max=0.4)

        export_gee_data(id, export_image_landsat, roi, 'l')


def export_gee_data(id, exportImage, roi, flag):
    """
    Export data from google earth engine to google drive
    :param exportImage:
    :param roi:
    :param flag:
    :param google_drive_folder:
    :return:
    """

    if flag=='v1':
        # Define the export parameters
        gee_export(exportImage, description='VIIRS_RGB_True_Export', fileNamePrefix= id + '-viirs_true_rgb',
                             region=roi, scale=500)

    elif flag=='v2':
        # Define the export parameters
        gee_export(exportImage, description='VIIRS_RGB_False_Export', fileNamePrefix= id + '-viirs_false_rgb',
                             region=roi, scale=500)

    elif flag=='l':
        gee_export(exportImage, description='Landsat', fileNamePrefix= id + '-landsat-truecolour',
                             region=roi, scale=30)

    elif flag=='t':
        gee_export(exportImage, description='Terra', fileNamePrefix= id + '-terra',
                             region=roi, scale=500)

    elif flag=='a':
        gee_export(exportImage, description='Aqua', fileNamePrefix= id + '-aqua',
                             region=roi, scale=500)

    elif flag=='g':
        gee_export(exportImage, description='GOES', fileNamePrefix= id + '-goes',
                             region=roi, scale=500)


def gee_export(exportImage, description, fileNamePrefix, region, scale):

    export_task = ee.batch.Export.image.toDrive(
        image=exportImage,
        description=description,
        folder=google_drive_folder,  # Change this to your preferred folder in Google Drive
        fileNamePrefix= fileNamePrefix,
        region=region,  # Define the region to export
        scale=scale,  # Scale in meters
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


def plot_plume(id, viirs, modis, tif_folder, plot_folder):
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

        if 'viirs_true' in tif_file:

            source = 'viirs-true'
            hotspot_plot = True
            plot_rgb(id, tif_file, viirs, modis, plot_folder, source, hotspot_plot)

        elif 'viirs_false' in tif_file:
            source = 'viirs_false'
            hotspot_plot = False
            plot_rgb(id, tif_file, viirs, modis, plot_folder, source, hotspot_plot)

        if 'landsat' in tif_file:

            source = 'landsat'
            hotspot_plot = True
            plot_rgb(id, tif_file, viirs, modis, plot_folder, source, hotspot_plot)

        if 'terra' in tif_file:

            source = 'terra'
            hotspot_plot = True
            plot_rgb(id, tif_file, viirs, modis, plot_folder, source, hotspot_plot)

        if 'aqua' in tif_file:

            source = 'aqua'
            hotspot_plot = True
            plot_rgb(id, tif_file, viirs, modis, plot_folder, source, hotspot_plot)

        if 'goes' in tif_file:

            source = 'goes'
            hotspot_plot = True
            plot_rgb(id, tif_file, viirs, modis, plot_folder, source, hotspot_plot)


def plot_rgb(id, tif_file, hotspots_v, hotspots_m, plot_folder, source, hotspot_plot):

    with rio.open(tif_file) as src:
        # Read the image data
        # img_data = src.read(1)
        b1 = src.read(1)
        b2 = src.read(2)
        b3 = src.read(3)

        rgb = np.dstack((b1, b2, b3))

        bounds = src.bounds

        # Plotting
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

        if hotspot_plot:
            plt.scatter(hotspots_v['longitude'], hotspots_v['latitude'], s=2, alpha=0.3, color='red',
                        label='VIIRS SNPP Hotspots')
            plt.scatter(hotspots_m['longitude'], hotspots_m['latitude'], s=1, alpha=0.3, color='orange',
                        label='MODIS Hotspots')
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
        id, coords, hotspot_no_days, hotspot_date, start_date, end_date = get_info(plume_excel_sheet.iloc[ii])

        print('obtaining hotspots')
        viirs_snpp, modis = obtain_hotspots(id, FIRMS_map_key, coords, hotspot_no_days, hotspot_date)

        print('accessing and downloading gee imagery to drive')
        # WARNING: The Landsat scale is currently set to export at 50m, and exporting the .tif takes quite a while!
        landsat_flag = 0
        #download_gee_data(id, coords, start_date, end_date, hotspot_date, landsat_flag)

        print('downloading data from drive to local path')
        #drive_download_data()

        #plot_plume(id, viirs_snpp, modis, download_path, plot_path)
        print('Completed fire ' + str(id))













