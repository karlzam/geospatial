import ee
import geemap
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
from cartopy import crs as ccrs
from geodatasets import get_path
import matplotlib.patches as mpatches

from osgeo import gdal
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')

MAP_KEY = 'e865c77bb60984ab516517cd4cdadea0'
#url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
#try:
#  df = pd.read_json(url,  typ='series')
#  display(df)
#except:
#  # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
#  print ("There is an issue with the query. \nTry in your browser: %s" % url)

path = get_path("naturalearth.land")
world = gpd.read_file(path)

# VIIRS_NOAA20_NRT Sept 23 2023
area_url_NOAA20 = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_NOAA20_NRT/-125.6,48.9,-103,60.1/1/2023-09-23'
df_NOAA20 = pd.read_csv(area_url_NOAA20)
gdf_NOAA20 = gpd.GeoDataFrame(
    df_NOAA20, geometry=gpd.points_from_xy(df_NOAA20.longitude, df_NOAA20.latitude), crs="EPSG:4326"
)

# S-NPP & NOAA-20 Sept 23 2023
# There's standard processing and NRT, but NRT was only available for 2024 so using NRT
area_url_SNPP = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_SNPP_SP/-125.6,48.9,-103,60.1/1/2023-09-23'
df_SNPP = pd.read_csv(area_url_SNPP)
gdf_SNPP = gpd.GeoDataFrame(
    df_SNPP, geometry=gpd.points_from_xy(df_SNPP.longitude, df_SNPP.latitude), crs="EPSG:4326"
)

# Get persistent hot spots (these are from Piyush)
persistent_hs = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\persistent_heat_sources\m3mask5_lcc.shp')
persistent_hs_sub = persistent_hs[(persistent_hs['prov']=='AB') | (persistent_hs['prov']=='BC') | (persistent_hs['prov']=='SK')]
persistent_hs_sub = persistent_hs_sub.to_crs(epsg=4326)

# Read in NBAC perimeters from 2023
NBAC = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp')
NBAC = NBAC.to_crs(epsg=4326)

# Restrict NBAC to these bounds -125.6,48.9,-103,60.1
bounds = NBAC.bounds

# x is negative
# y is positive

index_to_drop = []
for ii in range(0, len(NBAC)):
    if bounds.loc[ii]['minx'] < -125.6:
        index_to_drop.append(ii)
    elif bounds.loc[ii]['maxx'] > -103:
        index_to_drop.append(ii)
    elif bounds.loc[ii]['miny'] < 48.9:
        index_to_drop.append(ii)
    elif bounds.loc[ii]['maxy'] > 60.1:
        index_to_drop.append(ii)

#NBAC_sub = NBAC.copy()
#for ii in range(0, len(index_to_drop)):
#    NBAC_sub = NBAC_sub.drop(index_to_drop[ii])


#buffer_poly = gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\buffer-polygon-2.shp')
#buffer_distance = 30000
#gdf_projected_2 = buffer_poly.to_crs(CRS(32610))# 5 km in meters
#gdf_buffered_2 = gdf_projected_2.copy()
#gdf_buffered_2['geometry'] = gdf_buffered_2.buffer(buffer_distance)
#gdf_buffered_2 = gdf_buffered_2.to_crs(epsg=4326)
#gdf_buffered_2.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\buffer-polygon-large.shp')

#buffer_distance = 1000
#NBAC_sub_proj = NBAC_sub.to_crs(CRS(32610))
#NBAC_buffer = NBAC_sub_proj.copy()
#NBAC_buffer['geometry'] = NBAC_buffer.buffer(buffer_distance)
#NBAC_buffer = NBAC_buffer.to_crs(epsg=4326)
#NBAC_buffer.to_file(r'C:\Users\kzammit\Documents\FRP\NBAC-buffer.shp')

NBAC_buffer = gpd.read_file(r'C:\Users\kzammit\Documents\FRP\NBAC-buffer.shp')

# Get polygon of Canada for plotting
shp_map = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_admin_1_states_provinces.shp')
ab_bc_ma_map = shp_map[(shp_map['name']=='Alberta') | (shp_map['name']=='British Columbia') | (shp_map['name']=='Saskatchewan')]
ab_bc_ma_map.head()

# Plot
fig, ax = plt.subplots(figsize=(10,8))
ab_bc_ma_map.plot(ax=ax, edgecolor='black', linewidth=1, cmap='Greys')
# gdf_NOAA20.plot(ax=ax, color='red', markersize=15, label='NOAA20')
NBAC_buffer.plot(ax=ax, color='black', markersize=5, label='NBAC')
#NBAC.plot(ax=ax, color='yellow', markersize=5, label='NBAC')
gdf_NOAA20.plot(ax=ax, color='red', markersize=1, label='NOAA20')
gdf_SNPP.plot(ax=ax, color='orange', markersize=1, label='SNPP')

black = mpatches.Patch(color='black', label='NBAC')
ax.legend(handles=[black])

plt.title('NOAA20 vs SNPP Hot Spots from Sept 23 2023')
plt.savefig(r'C:\Users\kzammit\Documents\FRP\NOAA20-SNPP.png')
#plt.show()

print('test')

# Ideas
# One: explore FRP values
# Write a script to highlight detections which aren't within the NBAC perimeters
# See if there's any difference in FRP for those within perimeters vs outside of perimeters
