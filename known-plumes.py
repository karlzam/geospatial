import ee
import geemap
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from geodatasets import get_path
import rioxarray as rxr
import matplotlib.patches as mpatches
import time

from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

ee.Authenticate()
ee.Initialize(project='karlzam')

shp_map = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_admin_1_states_provinces.shp')

usa = shp_map[shp_map['admin']=='United States of America']



print('test')