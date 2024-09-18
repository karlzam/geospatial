import ee
import geemap
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Proj

from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

## Load basemap
# Need the shx file too
map = gpd.read_file(r'C:\Users\Karlee\Documents\NFC\Shapefiles\ne_50m_admin_1_states_provinces.shp')
ca_map = map[map['iso_a2']=='CA']
ab_bc_ma_map = ca_map[(ca_map['name']=='Alberta') | (ca_map['name']=='British Columbia') | (ca_map['name']=='Saskatchewan')]

# Get subset polygon of example fire
# This is in easting northing
# Current CRS is 3978 (NAD83_NRCan_LCC_Canada)
subset_poly = gpd.read_file(r'C:\Users\Karlee\Documents\NFC\Sep23-Fire\subset-polygon.shp')
subset_poly = gpd.GeoDataFrame(subset_poly)
#gdf = gdf.set_crs(epsg=3857)
subset_poly = subset_poly.to_crs(epsg=4326)

# Get csv of all Alberta fires from Sep 23
fires_sep23 = pd.read_csv(r'C:\Users\Karlee\Documents\NFC\Sep23-Fire\alberta-fires-sep23.csv')
df_fs23 = gpd.GeoDataFrame(fires_sep23, geometry=gpd.points_from_xy(fires_sep23.longitude, fires_sep23.latitude))

# Get persistent hot spots
# These are in easting, northing
persistent_hs = gpd.read_file(r'C:\Users\Karlee\Documents\NFC\Sep23-Fire\m3mask5_lcc.shp')

## Plot the background map
fig, ax = plt.subplots(figsize=(10,8))
ab_bc_ma_map.plot(ax=ax, edgecolor='black', linewidth=1, cmap='Greens')
# Set the background colour to blue
ax.set_facecolor('#a2d2ff')
# plot the subset polygon
subset_poly.plot(ax=ax)
# Plot the alberta fires from Sep 23 2023
df_fs23.plot(ax=ax, color='red', marker='*', markersize=10)
#persistent_hs.plot(ax=ax, color='black')
plt.show()

print('test')

