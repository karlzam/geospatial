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
import time
from glob import glob
import rasterio as rio
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import numpy as np
from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

ee.Authenticate()
ee.Initialize(project='karlzam')

###################################################################################################
# Variables
###################################################################################################

# Create a service key for your google cloud project and download the .json to your local machine
creds_file = r'C:\Users\kzammit\Documents\google-drive-API-key\karlzam-d3258a83d6cb.json'


###################################################################################################
# Methods
###################################################################################################


if __name__ == "__main__":

    print('test')












