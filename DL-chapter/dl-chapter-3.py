# Question two: can we classify VIIRS hotspots as T or F positive using a BNN?
# " we present a method for training a probabilistic BNN to perform classification on the vectorized fire data into
# the classes: true or false positive"
import math

# Step One: Pull all VIIRS hotspots for Canada for a specific day
# Step Two: Load NBAC, NFDB, and persistent hot spots
# Step Three: Buffer these by 375*sqrt(2)
# Step Four: Flag as outside or inside a known boundary and set this is as "0" and "1" for TP/FP
# Step Five: Apply to hotspots from a different day and classify as TP/FP

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
import shapefile as shp
import numpy as np
from shapely.geometry import box
import math


###### User Inputs ######

# Obtained from: https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/
nbac_shp = r'C:\Users\kzammit\Documents\shp\nbac\nbac_2023_20240530.shp'

# At the time of writing this script, NFDB polygons were not available for 2023
# Obtained from: https://cwfis.cfs.nrcan.gc.ca/datamart/download/nfdbpnt
nfdb_shp = r'C:\Users\kzammit\Documents\shp\nfdb\NFDB_point_20240613.shp'

# Obtained from Piyush
pers_hs_shp = r'C:\Users\kzammit\Documents\shp\pers-hs\m3mask5_lcc.shp'

# Obtained from https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/
# ne_10m_admin_0_countries.zip
nat_earth_shp = r'C:\Users\kzammit\Documents\shp\nat-earth\ne_10m_admin_0_countries.shp'

# Date to focus analysis on (training used 2023, val used 2022)
doi = '2023/09/23'
yoi = 2023

# FIRMS may key to use API
FIRMS_map_key = 'e865c77bb60984ab516517cd4cdadea0'
doi_firms = '2023-09-23'

###### Code ######

### NBAC
# Import all fires for 2023
# Obtained from: https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/
nbac = gpd.read_file(nbac_shp)

# NBAC has agency start date as well as hotspot start date, and these often do not align
# If one of them is missing, the fill value is "0000/00/00"
# Create start and end date columns from the earlier or later of the two respectively
nbac['start_date'] = 'tbd'

# If either agency or hotspot start date is empty, assign the other as the start date
nbac.loc[nbac['HS_SDATE'] == '0000/00/00', 'start_date'] = nbac['AG_SDATE']
nbac.loc[nbac['AG_SDATE'] == '0000/00/00', 'start_date'] = nbac['HS_SDATE']

# Pick the earlier of the two for the start date, excluding the empties
nbac.loc[(nbac['HS_SDATE'] <= nbac['AG_SDATE']) & (nbac['HS_SDATE'] != '0000/00/00'), 'start_date'] = nbac['HS_SDATE']
nbac.loc[(nbac['AG_SDATE'] <= nbac['HS_SDATE']) & (nbac['AG_SDATE'] != '0000/00/00'), 'start_date'] = nbac['AG_SDATE']

# Do the same steps for the end date
nbac['end_date'] = 'tbd'
nbac.loc[nbac['HS_EDATE'] == '0000/00/00', 'end_date'] = nbac['AG_EDATE']
nbac.loc[nbac['AG_EDATE'] == '0000/00/00', 'end_date'] = nbac['HS_EDATE']
nbac.loc[(nbac['HS_EDATE'] >= nbac['AG_EDATE']) & (nbac['HS_EDATE'] != '0000/00/00'), 'end_date'] = nbac['HS_EDATE']
nbac.loc[(nbac['AG_EDATE'] >= nbac['HS_EDATE']) & (nbac['AG_EDATE'] != '0000/00/00'), 'end_date'] = nbac['AG_EDATE']

# There are some cases where there is no agency date OR hotspot date, drop these
nbac = nbac.drop(nbac[(nbac.start_date == '0000/00/00')].index)
nbac = nbac.drop(nbac[(nbac.end_date == '0000/00/00')].index)

# Filter the hotspots so we're only looking at fires which contain Sept 23 within their date range
date_format = '%Y/%m/%d'
date_obj = datetime.strptime(doi, date_format)
nbac['doi'] = 0
nbac = nbac.assign(start_dt = lambda x: pd.to_datetime(x['start_date'], format=date_format))
nbac = nbac.assign(end_dt = lambda x: pd.to_datetime(x['end_date'], format=date_format))
nbac.loc[(nbac['start_dt'] <= date_obj) & (nbac['end_dt'] >= date_obj), "doi"] = 1

# Create a new dataframe for only fires containing the date of interest
nbac_doi = nbac[nbac['doi']==1]
nbac_doi = nbac_doi.reset_index()


### NFDB
nfdb = gpd.read_file(nfdb_shp)
nfdb_doi = nfdb[nfdb['YEAR']==yoi]
nfdb_doi = nfdb_doi.drop(nfdb_doi[(nfdb_doi.OUT_DATE == '0000/00/00')].index)
nfdb_doi['doi'] = 0
nfdb_doi = nfdb_doi.assign(start_dt = lambda x: pd.to_datetime(x['REP_DATE'], format=date_format))
nfdb_doi = nfdb_doi.assign(end_dt = lambda x: pd.to_datetime(x['OUT_DATE'], format=date_format))
nfdb_doi.loc[(nfdb_doi['start_dt'] <= date_obj) & (nfdb_doi['end_dt'] >= date_obj), "doi"] = 1
nfdb_doi = nfdb_doi[nfdb_doi['doi']==1]
nfdb_doi = nfdb_doi.reset_index()


### Persistent heat sources
pers_hs = gpd.read_file(pers_hs_shp)
cad_provs = ['NL', 'PE', 'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
pers_hs_cad = pers_hs[pers_hs['prov'].isin(cad_provs)]


### Apply buffers to all sounds to account for hotspot resolution
# NAD83 (EPSG 3978) is commonly used for Canada
# Helper function to project and buffer geometries
def project_and_buffer(gdf, target_epsg, buffer_dist):
    gdf_projected = gdf.to_crs(target_epsg)
    gdf_projected['geometry'] = gdf_projected['geometry'].buffer(buffer_dist)
    gdf_projected = gdf_projected.to_crs(epsg=4326)
    return gdf_projected

buff_nbac = {}
buff_nfdb = {}
buff_pers_hs = {}

# Buffer distance
buffer_distances = {
    "viirs": math.sqrt(2) * 375,
    "modis": math.sqrt(2) * 1000,
    "landsat": math.sqrt(2) * 30,
    "goes": math.sqrt(2) * 4000
}

target_epsg = 3978

# # This takes a bit to run, so we'll save the files and load them in individually after running once
# for sensor, buffer_dist in buffer_distances.items():
#     buff_nbac[sensor] = project_and_buffer(nbac_doi, target_epsg, buffer_dist)
#     buff_nfdb[sensor] = project_and_buffer(nfdb_doi, target_epsg, buffer_dist)
#     buff_pers_hs[sensor] = project_and_buffer(pers_hs_cad, target_epsg, buffer_dist)
#
dir = r'C:\Users\kzammit\Documents\DL-chapter\shp'
# buff_nbac['viirs'].to_file(dir + '\\' + 'nbac-buff-viirs.shp')
# buff_nbac['landsat'].to_file(dir + '\\' + 'nbac-buff-landsat.shp')
# buff_nbac['goes'].to_file(dir + '\\' + 'nbac-buff-goes.shp')
# buff_nbac['modis'].to_file(dir + '\\' + 'nbac-buff-modis.shp')
#
# buff_nfdb['viirs'].to_file(dir + '\\' + 'nfdb-buff-viirs.shp')
# buff_nfdb['landsat'].to_file(dir + '\\' + 'nfdb-buff-landsat.shp')
# buff_nfdb['goes'].to_file(dir + '\\' + 'nfdb-buff-goes.shp')
# buff_nfdb['modis'].to_file(dir + '\\' + 'nfdb-buff-modis.shp')
#
# buff_pers_hs['viirs'].to_file(dir + '\\' + 'pers_hs-buff-viirs.shp')
# buff_pers_hs['landsat'].to_file(dir + '\\' + 'pers_hs-buff-landsat.shp')
# buff_pers_hs['goes'].to_file(dir + '\\' + 'pers_hs-buff-goes.shp')
# buff_pers_hs['modis'].to_file(dir + '\\' + 'pers_hs-buff-modis.shp')
#

# # check the buffering was done correctly
# fig, ax = plt.subplots(figsize=(10,8))
# buff_nbac['viirs'].plot(ax=ax, color='red', linewidth=1)
# nbac_doi.plot(ax=ax, color='black', linewidth=1)
# plt.savefig('test-viirs.png')
#
# fig, ax = plt.subplots(figsize=(10,8))
# buff_nbac['goes'].plot(ax=ax, color='red', linewidth=1)
# nbac_doi.plot(ax=ax, color='black', linewidth=1)
# plt.savefig('test-goes.png')


buff_nbac['viirs'] = gpd.read_file(dir + '\\' + 'nbac-buff-viirs.shp')
buff_nbac['landsat'] = gpd.read_file(dir + '\\' + 'nbac-buff-landsat.shp')
buff_nbac['goes'] = gpd.read_file(dir + '\\' + 'nbac-buff-goes.shp')
buff_nbac['modis'] = gpd.read_file(dir + '\\' + 'nbac-buff-modis.shp')

buff_nfdb['viirs'] = gpd.read_file(dir + '\\' + 'nfdb-buff-viirs.shp')
buff_nfdb['landsat'] = gpd.read_file(dir + '\\' + 'nfdb-buff-landsat.shp')
buff_nfdb['goes'] = gpd.read_file(dir + '\\' + 'nfdb-buff-goes.shp')
buff_nfdb['modis'] = gpd.read_file(dir + '\\' + 'nfdb-buff-modis.shp')

buff_pers_hs['viirs'] = gpd.read_file(dir + '\\' + 'pers_hs-buff-viirs.shp')
buff_pers_hs['landsat'] = gpd.read_file(dir + '\\' + 'pers_hs-buff-landsat.shp')
buff_pers_hs['goes'] = gpd.read_file(dir + '\\' + 'pers_hs-buff-goes.shp')
buff_pers_hs['modis'] = gpd.read_file(dir + '\\' + 'pers_hs-buff-modis.shp')

nbac_doi = nbac_doi.to_crs(epsg=4326)
nfdb_doi = nfdb_doi.to_crs(epsg=4326)
pers_hs_cad = pers_hs_cad.to_crs(epsg=4326)

# Join them all into a single dataframe (it's ok that it's messy, we'll just be using the geometry column)
# The epsg code are already the same for all (defined in the function)
#df_perims = pd.concat([nbac_buff, nfdb_buff, pers_hs_cad_buff])

df_perims= {
    "viirs": pd.concat([buff_nbac['viirs'], buff_nfdb['viirs'], buff_pers_hs['viirs']]),
    "modis": pd.concat([buff_nbac['modis'], buff_nfdb['modis'], buff_pers_hs['modis']]),
    "landsat": pd.concat([buff_nbac['landsat'], buff_nfdb['landsat'], buff_pers_hs['landsat']]),
    "goes": pd.concat([buff_nbac['goes'], buff_nfdb['goes'], buff_pers_hs['goes']])
}

### Canada
ne = gpd.read_file(nat_earth_shp)
cad = ne[ne['ADMIN']=='Canada']

# Create a bounding box around Canada (this will include some of the States, but we'll fix this later)
bbox_coords = cad.bounds
bbox_coords = bbox_coords.reset_index()
coords = f"{bbox_coords['minx'][0]},{bbox_coords['miny'][0]},{bbox_coords['maxx'][0]},{bbox_coords['maxy'][0]}"


### Hotspots
# Only SNPP and NOAA20 are available for 2023 (NOAA21 came after)

# FIRMS
MAP_KEY = FIRMS_map_key
url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
try:
    df = pd.read_json(url, typ='series')
except:
    # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
    print("There is an issue with the query. \nTry in your browser: %s" % url)


coords_str = str(coords) + '/' + str(1) + '/' + str(doi_firms)
url_SNPP = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_SP/{coords_str}'
url_NOAA20 = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_NOAA20_NRT/{coords_str}'
url_MODIS = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/MODIS_SP/{coords_str}'
url_GOES = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/GOES_NRT/{coords_str}'
url_landsat = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/LANDSAT_NRT/{coords_str}'


df_SNPP = pd.read_csv(url_SNPP)
gdf_SNPP = gpd.GeoDataFrame(
    df_SNPP, geometry=gpd.points_from_xy(df_SNPP.longitude, df_SNPP.latitude), crs="EPSG:4326"
)

df_NOAA20 = pd.read_csv(url_NOAA20)
gdf_NOAA20 = gpd.GeoDataFrame(
    df_NOAA20, geometry=gpd.points_from_xy(df_NOAA20.longitude, df_NOAA20.latitude), crs="EPSG:4326"
)

df_MODIS = pd.read_csv(url_MODIS)
gdf_MODIS = gpd.GeoDataFrame(
    df_MODIS, geometry=gpd.points_from_xy(df_MODIS.longitude, df_MODIS.latitude), crs="EPSG:4326"
)

df_GOES = pd.read_csv(url_GOES)
gdf_GOES = gpd.GeoDataFrame(
    df_GOES, geometry=gpd.points_from_xy(df_GOES.longitude, df_GOES.latitude), crs="EPSG:4326"
)

df_landsat = pd.read_csv(url_landsat)
gdf_landsat = gpd.GeoDataFrame(
    df_landsat, geometry=gpd.points_from_xy(df_landsat.longitude, df_landsat.latitude), crs="EPSG:4326"
)

# Remove hotspots outside of Canada
gdf_SNPP = gpd.sjoin(gdf_SNPP, cad, predicate='within', how='left')
gdf_SNPP = gdf_SNPP[gdf_SNPP.index_right.notnull()].copy()
gdf_SNPP = gdf_SNPP.drop(gdf_SNPP.columns[16:], axis=1)

gdf_NOAA20 = gpd.sjoin(gdf_NOAA20, cad, predicate='within', how='left')
gdf_NOAA20 = gdf_NOAA20[gdf_NOAA20.index_right.notnull()].copy()
gdf_NOAA20= gdf_NOAA20.drop(gdf_NOAA20.columns[15:], axis=1)

gdf_MODIS = gpd.sjoin(gdf_MODIS, cad, predicate='within', how='left')
gdf_MODIS = gdf_MODIS[gdf_MODIS.index_right.notnull()].copy()
gdf_MODIS = gdf_MODIS.drop(gdf_MODIS.columns[16:], axis=1)


gdf_GOES = gpd.sjoin(gdf_GOES, cad, predicate='within', how='left')
gdf_GOES = gdf_GOES[gdf_GOES.index_right.notnull()].copy()
gdf_GOES= gdf_GOES.drop(gdf_GOES.columns[16:], axis=1)

gdf_landsat = gpd.sjoin(gdf_landsat, cad, predicate='within', how='left')
gdf_landsat = gdf_landsat[gdf_landsat.index_right.notnull()].copy()
gdf_landsat= gdf_landsat.drop(gdf_landsat.columns[16:], axis=1)

gdf_VIIRS = pd.concat([gdf_SNPP, gdf_NOAA20])

# Dictionary to store the balanced results for each source type
balanced_results = {}

# Loop through each source type in df_perims
for source_type, perimeters in df_perims.items():
    # Filter the hotspots for the current source type
    if source_type == "viirs":
        source_hotspots = gdf_VIIRS
    elif source_type == "modis":
        source_hotspots = gdf_MODIS
    elif source_type == "landsat":
        source_hotspots = gdf_landsat
    elif source_type == "goes":
        source_hotspots = gdf_GOES
    else:
        continue  # Skip unsupported source types

    # Perform the spatial join
    joined = gpd.sjoin(source_hotspots, perimeters, predicate='within', how='left')

    # Clean up the dataframe
    joined = joined.drop(joined.columns[19:], axis=1, errors='ignore')  # Adjust this index based on your dataset
    joined['Class'] = 1
    joined.loc[joined.index_right0.isnull(), 'Class'] = 0  # Update the 'Class' column for hotspots outside perimeters
    joined = joined.drop(['index', 'index_right', 'type', 'geometry'], axis=1, errors='ignore')  # Adjust columns as needed

    # Balance the dataset by randomly sampling
    num_fp = (joined['Class'] == 0).sum()
    fp_df = joined[joined['Class'] == 0]

    num_tp = (joined['Class'] == 1).sum()
    tp_df = joined[joined['Class'] == 1]

    if num_tp > num_fp:
        fp_sample = fp_df.sample(n=num_fp, random_state=42)
        tp_sample = tp_df.sample(n=num_fp, random_state=42)
    elif num_fp > num_tp:
        fp_sample = fp_df.sample(n=num_tp, random_state=42)
        tp_sample = tp_df.sample(n=num_tp, random_state=42)
    else:
        # If they are already balanced
        fp_sample = fp_df
        tp_sample = tp_df

    # Combine the balanced data
    df_balanced = pd.concat([fp_sample, tp_sample])

    # Drop non-numeric columns
    df_balanced = df_balanced.drop(['acq_date', 'satellite', 'instrument', 'version'], axis=1, errors='ignore')

    # Encode categorical columns
    df_balanced['daynight'] = df_balanced['daynight'].replace({'D': 1, 'N': 0})
    df_balanced['confidence'] = df_balanced['confidence'].replace({'l': 0, 'n': 1, 'h': 2})

    # Store the processed and balanced dataframe in the results dictionary
    balanced_results[source_type] = df_balanced

# Initialize an empty list to store dataframes with source labels
labeled_dfs = []

# Mapping of source types to numeric labels
source_labels = {"viirs": 0, "modis": 1, "landsat": 3, "goes": 2}

# Loop through each source type and its balanced dataframe
for source_type, df_balanced in balanced_results.items():
    # Add a new column to indicate the source
    df_balanced['source'] = source_labels[source_type]

    # Append the labeled dataframe to the list
    labeled_dfs.append(df_balanced)

# Concatenate all labeled dataframes into a single dataframe
final_df = pd.concat(labeled_dfs, ignore_index=True)
final_df = final_df.drop(['index_right0'], axis=1, errors='ignore')

# Drop landsat because it doesn't have frp
#final_df = final_df.drop(final_df[final_df['source'] == 3].index, inplace = True)

# Drop all rows containing any nans (TODO: Is there a way to keep brightness values?)
final_df = final_df.drop(final_df.columns[13:], axis=1)
final_df = final_df.drop(['YEAR'], axis=1)
final_df = final_df.dropna()

# Save the final dataframe to an Excel file
final_df.to_excel(r'C:\Users\kzammit\Documents\DL-chapter\train-q3.xlsx', index=False)







