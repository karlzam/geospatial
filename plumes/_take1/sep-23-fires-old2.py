# This script plots all fires reported through NBAC and the NFDB for Sep 22/23 2023.
#
#

# =================================================================================
# Imports
# =================================================================================

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
import shapefile as shp
import numpy as np


########## NBAC ##########

# hs_sdate: date of the first detected hotspot within the spatial extent of the fire event
# hs_edate: date of the last detected hotspot within the spatial extent of the fire event
# ag_sdate: fire start date reported by the agency
# ag_edate: end date reported by the agency
# capdate: acquisition date of the source data
# poly_ha: total area calculated in hectares (canada albers equal area conic projection)
# adj_ha: adjusted area burn calculated in hectares
# gid: fire year and NFIREID concat
NBAC = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\NBAC\nbac_2023_20240530.shp')
#
date_format = '%Y/%m/%d'

# The hotspot and agency start/end dates do not always align, so need to make if statement and grab the earlier
# of the two for the start date and the later of the two for the end date, while also accounting for the '0000/00/00'
# if there was no date reported (I'm assuming)
NBAC['start_date'] = 'tbd'
NBAC.loc[NBAC['HS_SDATE'] == '0000/00/00', 'start_date'] = NBAC['AG_SDATE']
NBAC.loc[NBAC['AG_SDATE'] == '0000/00/00', 'start_date'] = NBAC['HS_SDATE']
NBAC.loc[(NBAC['HS_SDATE'] <= NBAC['AG_SDATE']) & (NBAC['HS_SDATE'] != '0000/00/00'), 'start_date'] = NBAC['HS_SDATE']
NBAC.loc[(NBAC['AG_SDATE'] <= NBAC['HS_SDATE']) & (NBAC['AG_SDATE'] != '0000/00/00'), 'start_date'] = NBAC['AG_SDATE']

NBAC['end_date'] = 'tbd'
NBAC.loc[NBAC['HS_EDATE'] == '0000/00/00', 'end_date'] = NBAC['AG_EDATE']
NBAC.loc[NBAC['AG_EDATE'] == '0000/00/00', 'end_date'] = NBAC['HS_EDATE']
NBAC.loc[(NBAC['HS_EDATE'] >= NBAC['AG_EDATE']) & (NBAC['HS_EDATE'] != '0000/00/00'), 'end_date'] = NBAC['HS_EDATE']
NBAC.loc[(NBAC['AG_EDATE'] >= NBAC['HS_EDATE']) & (NBAC['AG_EDATE'] != '0000/00/00'), 'end_date'] = NBAC['AG_EDATE']

# There are some cases where there is no agency date OR hotspot date
# Drop these
NBAC = NBAC.drop(NBAC[(NBAC.start_date == '0000/00/00')].index)
NBAC = NBAC.drop(NBAC[(NBAC.end_date == '0000/00/00')].index)

# Filter the hotspots so that we're only looking at fires which contain Sept 23 within their date range
date_obj = datetime.strptime('2023/09/23', date_format)
NBAC['sept_23'] = 0
NBAC = NBAC.assign(start_dt = lambda x: pd.to_datetime(x['start_date'], format=date_format))
NBAC = NBAC.assign(end_dt = lambda x: pd.to_datetime(x['end_date'], format=date_format))

# This is the line throwing the warning, redid with their suggestion
#NBAC['sept_23'][(NBAC['start_dt'] <= date_obj) & (NBAC['end_dt'] >= date_obj)] = 1
NBAC.loc[(NBAC['start_dt'] <= date_obj) & (NBAC['end_dt'] >= date_obj), "sept_23"] = 1

NBAC['full-year'] = 0
date_obj_s = datetime.strptime('2023/01/01', date_format)
date_obj_e = datetime.strptime('2023/12/31', date_format)
NBAC.loc[(NBAC['start_dt'] >= date_obj_s) & (NBAC['end_dt'] <= date_obj_e), "full-year"] = 1

# Create a dataframe for just those fires containing Sept 23 2023
NBAC_Sep23 = NBAC[NBAC['sept_23']==1]
NBAC_Sep23 = NBAC_Sep23.reset_index()

NBAC_all = NBAC[NBAC['full-year']==1]
NBAC_all = NBAC_all.reset_index()
# # Apply 375*3 buffer to NBAC fires
buffer_distance = 375*3
# NBAC_buffered = NBAC_Sep23.copy()
# NBAC_buffered['geometry'] = NBAC_buffered.buffer(buffer_distance)
# NBAC_buffered.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\buffered-sept23.shp')
NBAC23_buffered = gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\buffered-sept23.shp')

# # Apply 375*3 buffer to NBAC fires
#NBAC_all_buffered = NBAC_all.copy()
#NBAC_all_buffered['geometry'] = NBAC_all_buffered.buffer(buffer_distance)
#NBAC_all_buffered.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\NBAC-all-buffered.shp')
NBAC_all_buffered = gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\NBAC-all-buffered.shp')


########## NFDB ##########

# # Apply buffer to NFDB point sources
# = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\CWFIS-NFDB\NFDB_point_20240613.shp')
#NFDB_23 = NFDB[NFDB['YEAR']==2023]
# # Drop the '0000' rows - this will be omitting data
# NFDB_23 = NFDB_23.drop(NFDB_23[(NFDB_23.OUT_DATE == '0000/00/00')].index)
# NFDB_23['sept_23'] = 0
# NFDB_23 = NFDB_23.assign(start_dt = lambda x: pd.to_datetime(x['REP_DATE'], format=date_format))
# NFDB_23 = NFDB_23.assign(end_dt = lambda x: pd.to_datetime(x['OUT_DATE'], format=date_format))
# NFDB_23.loc[(NFDB_23['start_dt'] <= date_obj) & (NFDB_23['end_dt'] >= date_obj), "sept_23"] = 1
# NFDB_0923 = NFDB_23[NFDB_23['sept_23']==1]


#NFDB_buffered = NFDB_0923.copy()
#NFDB_buffered['geometry'] = NFDB_buffered.buffer(buffer_distance)
#NFDB_buffered.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\NFDB-buffered-sept23.shp')
NFDB_buffered = gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\NFDB-buffered-sept23.shp')

########## Persistent Heat Sources ##########

# Add persistent heat sources
persistent_hs = gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\m3mask5_lcc.shp')
cad_provs = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
persistent_hs_cad = persistent_hs[persistent_hs['prov'].isin(cad_provs)]

# Add buffer to persistent heat source locations
#PHS_buffered = persistent_hs_cad.copy()
#PHS_buffered['geometry'] = PHS_buffered.buffer(buffer_distance)
#PHS_buffered.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\PHS-buffered.shp')
PHS_buffered= gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\PHS-buffered.shp')
persistent_hs_proj = PHS_buffered.to_crs(epsg=4326)

########## Hot Spots ##########

# Get box of coordinates for Canada to pull hotspots from FIRMS
# This will also pull some hotspots from below the border
canada = shp.Reader(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\canada.shp')
coords = str(canada.bbox[0]) + ',' + str(canada.bbox[1]) + ',' + str(canada.bbox[2]) + ',' + str(canada.bbox[3])

# Now we have all reported fires for Sept 23 2023 in Canada
# Now get list of hotspots for this day in Canada
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'
MAP_KEY = FIRMS_map_key
url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
try:
    df = pd.read_json(url, typ='series')
except:
    # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
    print("There is an issue with the query. \nTry in your browser: %s" % url)

## VIIRS_S-NPP
area_url_SNPP = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_SNPP_SP/' +
                 str(coords) + '/' + str(1) + '/' + str('2023-09-23'))
df_SNPP = pd.read_csv(area_url_SNPP)

gdf_SNPP = gpd.GeoDataFrame(
    df_SNPP, geometry=gpd.points_from_xy(df_SNPP.longitude, df_SNPP.latitude), crs="EPSG:4326"
)

## VIIRS_NOAA20
area_url_NOAA20 = ('https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + '/VIIRS_NOAA20_NRT/' +
                 str(coords) + '/' + str(1) + '/' + str('2023-09-23'))
df_NOAA20 = pd.read_csv(area_url_NOAA20)

gdf_NOAA20 = gpd.GeoDataFrame(
    df_NOAA20, geometry=gpd.points_from_xy(df_NOAA20.longitude, df_NOAA20.latitude), crs="EPSG:4326"
)

# Remove hotspots outside of Canada by performing a spatial join
map = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_admin_1_states_provinces.shp')
ca_map = map[map['iso_a2']=='CA']

gdf_SNPP = gpd.sjoin(gdf_SNPP, ca_map, predicate='within', how='left')
gdf_SNPP = gdf_SNPP[gdf_SNPP.index_right.notnull()].copy()
gdf_SNPP= gdf_SNPP.drop(gdf_SNPP.columns[17:], axis=1)

gdf_NOAA20 = gpd.sjoin(gdf_NOAA20, ca_map, predicate='within', how='left')
gdf_NOAA20 = gdf_NOAA20[gdf_NOAA20.index_right.notnull()].copy()
gdf_NOAA20= gdf_NOAA20.drop(gdf_NOAA20.columns[17:], axis=1)

########## IDENTIFY HOTSPOTS OUTSIDE OF KNOWN AREAS ##########

# Boundary sources: NFDB, NBAC, PHS
# Hotspot sources: SNPP, NOAA20

# Messy dataframe containing all sources
# Need to make sure they're all in the same projection first
NBAC23_proj = NBAC23_buffered.to_crs(epsg=4326)
NFDB = NFDB_buffered.to_crs(epsg=4326)
PHS = PHS_buffered.to_crs(epsg=4326)
df_all_known_23 = pd.concat([NBAC23_proj, NFDB, PHS])

NBAC_all_proj = NBAC_all_buffered.to_crs(epsg=4326)
df_all_known = pd.concat([NBAC_all_proj, NFDB, PHS])

# The hotspots are already in EPSG 4326
gdf_SNPP['sat'] = 'SNPP'
gdf_NOAA20['sat'] = 'NOAA20'
df_hotspots = pd.concat([gdf_SNPP, gdf_NOAA20])
df_hotspots = df_hotspots.drop(['index_right'], axis=1)

# Perform a spatial join to find hotspots within known geometries
# how returns the geometry column for only the specified dataframe
joined = gpd.sjoin(df_hotspots, df_all_known_23, predicate='within', how='left')

# Select only the rows that are not within any known geometries
df_outside_buffer = joined[joined.index_right0.isnull()].copy()

# Drop unnecessary columns without modifying in place
df_ob_clean_23 = df_outside_buffer.drop(df_outside_buffer.columns[17:], axis=1)

joined_all = gpd.sjoin(df_hotspots, df_all_known, predicate='within', how='left')
df_outside_buffer_all = joined_all[joined_all.index_right0.isnull()].copy()
df_ob_clean_all = df_outside_buffer_all.drop(df_outside_buffer_all.columns[17:], axis=1)

########## FLAG POINTS WITHIN 20KM OF URBAN AREA ##########

urban_areas = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_urban_areas.shp')
urban_areas = gpd.sjoin(urban_areas, ca_map, predicate='within', how='left')
urban_areas = urban_areas[urban_areas.index_right.notnull()].copy()
urban_areas = urban_areas.drop(urban_areas.columns[5:], axis=1)

# projected coord system for canada: ESRI:102001
#urban_areas_buffered = urban_areas.copy()
#urban_areas_buffered = urban_areas_buffered.to_crs('ESRI:102001')
#urban_areas_buffered['geometry'] = urban_areas_buffered.buffer(buffer_distance)
#urban_areas_buffered = urban_areas_buffered.to_crs(epsg=4326)
#urban_areas_buffered.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\urban-areas-buffered.shp')
urban_areas_buffered = gpd.read_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\urban-areas-buffered.shp')
urban_areas_buffered = urban_areas_buffered.to_crs(epsg=4326)

# keep hotspots, add flag for if there is an urban area
ua_hs_all = gpd.sjoin(df_ob_clean_all, urban_areas_buffered, predicate='within', how='left')
ua_hs_all['urban'] = np.where(~ua_hs_all['index_right'].isnull(),1,0)
ua_hs_all = ua_hs_all.drop(['index_right', 'scalerank_', 'featurecla', 'area_sqkm_', 'min_zoom_l'], axis=1)
print('For all 2023 boundaries, there are a total of ' + str(len(df_ob_clean_all)) +
      ' hotspots outside of the boundaries and ' + str(sum(ua_hs_all['urban'])) +
      ' of these are within urban areas.')

#ua_hs_all.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\all-hotspots-oob-2023.shp')
#ua_hs_all.to_excel(r'C:\Users\kzammit\Documents\Sept23-Fire\all-hotspots-oob-2023.xlsx', index=False)

# keep hotspots, add flag for if there is an urban area
ua_hs_23 = gpd.sjoin(df_ob_clean_23, urban_areas_buffered, predicate='within', how='left')
ua_hs_23['urban'] = np.where(~ua_hs_23['index_right'].isnull(),1,0)
ua_hs_23 = ua_hs_23.drop(['index_right', 'scalerank_', 'featurecla', 'area_sqkm_', 'min_zoom_l'], axis=1)
print('For boundaries with overlap on Sept 23, there are a total of ' + str(len(df_ob_clean_23)) +
      ' hotspots outside of the boundaries and ' + str(sum(ua_hs_23['urban'])) +
      ' of these are within urban areas.')

#ua_hs_23.to_file(r'C:\Users\kzammit\Documents\Sept23-Fire\shapefiles\all-hotspots-oob-sep23-2023.shp')
#ua_hs_23.to_excel(r'C:\Users\kzammit\Documents\Sept23-Fire\all-hotspots-oob-sep23-2023.xlsx', index=False)

########## PLOT ##########

# Plot the background map
map = gpd.read_file(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\ne_10m_admin_1_states_provinces.shp')
ca_map = map[map['iso_a2']=='CA']
# ca_map.to_file(r'C:\Users\kzammit\Documents\Shapefiles\Natural-Earth\canada.shp')

ca_map_proj = ca_map.to_crs(epsg=4326)
gdf_SNPP_proj = gdf_SNPP.to_crs(epsg=4326)
gdf_NOAA20_proj = gdf_NOAA20.to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(10,8))

ca_map_proj.plot(ax=ax, edgecolor='white', linewidth=1, color ='Black')

#gdf_SNPP_proj.plot(ax=ax, color='green', linewidth=1, markersize=1)
SNPP_patch = mpatches.Patch(color='green', label='SNPP-SP')

#gdf_NOAA20_proj.plot(ax=ax, color='purple', linewidth=1, markersize=1)
NOAA20_patch = mpatches.Patch(color='purple', label='NOAA20-NRT')

NBAC23_proj.plot(ax=ax, color='red', linewidth=1)
NBAC_patch = mpatches.Patch(color='red', label='NBAC')

NFDB.plot(ax=ax, color='white', linewidth=200, markersize=200)
NFDB_patch = mpatches.Patch(color='white', label='NFDB')

persistent_hs_proj.plot(ax=ax, color='deepskyblue', linewidth=1)
PSH_patch = mpatches.Patch(color='deepskyblue', label='Persistent HS')

urban_areas_buffered.plot(ax=ax, color='chartreuse', linewidth=1)
ua_patch = mpatches.Patch(color='chartreuse', label='Urban Areas')

#df_ob_clean_23.plot(ax=ax, color='yellow', linewidth=1, markersize=1)
df_ob_clean_patch_23 = mpatches.Patch(color='yellow', label='Hotspots Outside Borders Sept 23')

df_ob_clean_all.plot(ax=ax, color='yellow', linewidth=1, markersize=1)
df_ob_clean_patch_all = mpatches.Patch(color='yellow', label='Hotspots Outside Borders All 2023')

#plt.legend(handles=[NBAC_patch, NFDB_patch, PSH_patch, SNPP_patch, NOAA20_patch,
#                    df_ob_clean_patch_23, df_ob_clean_patch_all, ua_patch],
#           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.legend(handles=[NBAC_patch, NFDB_patch, PSH_patch, df_ob_clean_patch_all, ua_patch],
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Fires Active Sept 23 2023')
plt.savefig(r'C:\Users\kzammit\Documents\Sept23-Fire\sept-23-fires-ob.png', bbox_inches='tight')


